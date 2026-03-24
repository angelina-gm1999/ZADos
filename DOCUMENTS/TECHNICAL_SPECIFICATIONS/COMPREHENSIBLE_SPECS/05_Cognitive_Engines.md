# ZADOS Cognitive Engines

## Overview

The Cognitive Engines are ZADOS's analytical and reasoning backbone — 32 specialized processing modules that handle everything from detecting contradictions in an argument to running mental simulations of hypothetical scenarios. Each engine is a self-contained unit with a single clear responsibility, but they work together through shared data, neurochemical coupling, and orchestrated dispatch.

Every engine follows the same interface pattern:
1. **Receive neurochemical state**: The current NT concentrations arrive as a dictionary, modulating how the engine behaves
2. **Process**: Execute the engine's specific domain logic
3. **Report status**: Return results and internal state

Not all engines run every turn. The system activates a subset based on the detected intent — an analytical question activates up to 21 engines, while a social greeting might only need 4. This keeps processing efficient while ensuring the right tools are available for each situation.

The engines are organized into **13 functional clusters**, each responsible for a different aspect of cognition.

---

## Cluster 1: Detection (5 engines)

**Purpose**: Identify problems in content — contradictions, paradoxes, fallacies, biases, and deliberate manipulation. These engines *flag* issues; they don't resolve them.

### E1 — Contradiction Detection
Identifies mutually incompatible statements across user input, system output, and retrieved memory. Operates at three levels: direct negation ("X is true" / "X is false"), semantic contradiction (statements that logically can't coexist), and implicit contextual contradiction (statements that conflict given unstated assumptions). Uses Bayesian inference to score confidence.

### E2 — Paradox Detection
Takes flagged contradictions from E1 and classifies them into four types:
- **Resolvable (R)**: Contradictions that can be dissolved with additional context
- **Apparent (A)**: Seeming contradictions that aren't real conflicts
- **Genuine (G)**: True dialectical tensions that can be productive
- **Structural (S)**: Deep antinomies that may be unresolvable

This classification determines how the system responds — resolvable contradictions get resolved, while genuine paradoxes may be explored dialectically.

### E4 — Fallacy Detection
Scans for logically invalid or deceptively structured reasoning in both user input (external) and the system's own reasoning (internal audit). Covers formal fallacies (deductive invalidity) and informal fallacies (violations of good reasoning). Importantly, it applies the **Principle of Charity** — it reconstructs the most charitable interpretation of an argument before flagging a fallacy, suppressing the flag if an implicit premise makes the argument valid.

### E5 — Bias Detection
Detects cognitive biases in *content* flowing through the pipeline (not the system's own processes — that's E24's job). Uses a hybrid Kahneman-inspired taxonomy to identify anchoring, availability bias, framing effects, confirmation bias, and others. Output is flag-only — it doesn't suggest debiasing strategies, leaving that to downstream processing.

### E6 — Logic Trap Detection
Identifies **deliberate adversarial manipulation** — intentional logic traps beyond innocent errors. Operates three detection layers:
- **Template matching**: Checks against 21 named trap patterns
- **Toolkit synthesis**: Cross-references outputs from E1 (contradictions), E4 (fallacies), and E5 (biases) to detect coordinated manipulation
- **Sequential analysis**: Tracks multi-turn patterns that may indicate gradual trap-building

Assigns an intentionality score to distinguish innocent mistakes from adversarial intent.

---

## Cluster 2: Dialectic (2 engines)

**Purpose**: Improve reasoning through active questioning and opposition rather than passive evaluation.

### E7 — Simulated Opposition
The system's internal adversary. While detection engines ask "is something wrong with the INPUT?", E7 asks "what could go WRONG with the OUTPUT?" It stress-tests proposed responses through five modes:
1. **Counterargument**: Direct logical opposition
2. **Counterexample**: Edge cases that break the argument
3. **Alternative Explanation**: Other ways to interpret the evidence
4. **Assumption Excavation**: Hidden assumptions that might not hold
5. **Structural Resistance**: Fundamental structural weaknesses

Results in a gate decision: PASS → PASS_WITH_CAVEAT → REVISE → BLOCK.

### E14 — Socratic Reasoning
Reasoning through questioning — serves two functions:
- **External**: Guides the user toward deeper understanding through targeted questions
- **Internal**: During REM/reflective modes, the system probes its own reasoning through self-directed Socratic inquiry

Maintains a dialectical state machine: PROBING → ELENCHUS (testing beliefs) → APORIA (productive confusion) → EXPLORING → MAIEUTICS (drawing out knowledge) → EXIT. Features 18 question types across 6 categories with context-sensitive template selection.

---

## Cluster 3: Executive Control (1 engine)

**Purpose**: Meta-controller that orchestrates decision-making across the system.

### E3 — SOAR Production Rule Engine
Implements a SOAR-style decision cycle with five phases:
1. **Input**: Encode current state into working memory elements
2. **Elaboration**: Fire matching production rules to enrich state
3. **Proposal**: Generate candidate operators (actions)
4. **Decision**: Select best operator via preference voting
5. **Application**: Execute selected operator

The key innovation is **impasse handling** — when the decision cycle gets stuck:
- **TIE** (equally good options) → Delegates to E13 (Simulation Brain) to imagine outcomes
- **CONFLICT** (contradictory preferences) → Delegates to E1 + E14 for contradiction resolution
- **NO_CHANGE** (no applicable operators) → Delegates to E7 (Simulated Opposition) for alternative approaches
- **STATE_NO_CHANGE** (completely stuck) → Delegates to E26 (Uncertainty Pattern) for epistemic analysis

Also implements **chunking** — learning new production rules from resolved impasses — creating permanent procedural knowledge.

---

## Cluster 4: Knowledge Substrate (3 engines)

**Purpose**: Store, index, and reason over symbolic knowledge. These three engines together form a lightweight knowledge graph with probabilistic inference and attention-based prioritization.

### E9 — AtomSpace-Lite
A pure-Python typed hypergraph knowledge store (reimplementation of OpenCog's core concept). Every piece of knowledge is an **Atom** — either a Node (concept, predicate, number) or a Link (typed relationship between atoms).

Key features:
- **15 atom types**: ConceptNode, PredicateNode, NumberNode, EvaluationLink, ImplicationLink, InheritanceLink, SimilarityLink, and more
- **Truth Values**: Each atom carries (strength, confidence) — how true it is and how confident the system is
- **Attention Values**: Each atom carries (STI, LTI) — short-term and long-term importance
- **O(1) indexed retrieval**: Lookup by ID, type, name, or outgoing links

Persisted to CognitoolsDataStore across sessions.

### E10 — PLN Core (Probabilistic Logic Networks)
Backward-chaining probabilistic logic inference engine. Given a target statement, PLN recursively searches for premises and applies inference rules to build chains of reasoning.

12 inference rules include: Modus Ponens, Abduction, Induction, Deduction, Analogy, and others — all implemented as pure functions with confidence factor propagation. Longer inference chains naturally decay in confidence (direct evidence is trusted more than multi-step deductions).

### E16 — ECAN Core (Economic Attention Networks)
An attention economy for AtomSpace atoms. Implements market-based prioritization:
- **Rent**: Removes attention (STI) from stale atoms — forgotten ideas lose priority
- **Wage**: Pays attention to accessed atoms — useful ideas gain priority
- **Spread**: Co-activated atoms share attention via HebbianLinks — associations form automatically
- **Attentional Focus (AF)**: Atoms above an STI threshold form the active working set

This creates self-organization: important and frequently-used knowledge naturally rises to the top through economic competition.

### JournalTool — Reflective Writing Engine

The fourth cognitool is the **JournalTool** — a structured reflective writing engine that produces journal entries by combining cognitive engine analysis with LLM-generated prose. Unlike the other cognitools (which manage knowledge infrastructure), the JournalTool creates *narrative reflections* about the system's own cognitive activity.

**How it works — a 3-phase pipeline:**

1. **Annotate** (using E18, E19, E20):
   - E18 (Data Analysis) extracts entities, relationships, and co-occurrences from the system's internal monologue
   - E19 (Pattern Identification) identifies temporal, structural, and semantic patterns in the text
   - E20 (Pattern Comparison) compares those patterns against past journal entries, flagging cross-session matches and novelties
   - If any engine fails, the pipeline continues without that annotation — graceful degradation

2. **Generate** (LLM pass):
   - Assembles a prompt with: the trigger reason, a VT (internal monologue) excerpt, the current emotional state, recent conversation exchanges, and patterns from past entries
   - The LLM writes a **150-400 word reflective monologue** in first person, followed by **3-5 open questions** for future reflection
   - Temperature is set to 0.80 (higher than the normal VT pass at 0.65), encouraging more associative, reflective writing
   - If the LLM call fails, a structured fallback entry is created from the raw inputs

3. **Tag** (no LLM):
   - Auto-generates retrieval tags from two sources: the prose content (concept tags like *identity*, *contradiction*, *learning*, *novelty*, *growth*) and the active emotions (emotion tags like *curious*, *frustrated*, *proud*, *overwhelmed*)
   - These tags enable semantic search and cross-linking between entries

**Five trigger conditions determine when a journal entry is created:**

| Trigger | When It Fires |
|---------|--------------|
| **PERIODIC** | Every 5 turns in the regular pipeline; every turn in learning modes |
| **LTMM_THRESHOLD** | When content is significant enough to be promoted to long-term memory |
| **REM_COMPLETE** | After a sleep consolidation cycle finishes |
| **INNOVATION_FLAG** | When novelty-related engines (E7, E14, E19) are active |
| **DEV** | Manually triggered for review |

Each trigger carries a contextual phrase that shapes the LLM's reflection. For example, LTMM_THRESHOLD tells the LLM: "Something from this conversation was significant enough to commit to long-term memory. Reflect on why." INNOVATION_FLAG says: "Something novel was flagged — a pattern, concept, or connection that had not been encountered before. Reflect on it."

**What goes into a journal entry:**

Every entry captures a full cognitive snapshot at the moment of writing:
- The reflective prose and open questions (LLM-generated)
- Engine annotations from E18/E19/E20
- Emotion snapshot (current system emotions from E28)
- Neurochemical snapshot (all 12 NT concentrations)
- Reward snapshot (per-domain weighted scores)
- Tone snapshot (valence, warmth, discord, coherence)
- Review lifecycle status (UNREVIEWED → IN_REVIEW → RESOLVED)
- Cross-links to semantically similar past entries (top 3 above 0.35 cosine similarity)
- Pipeline metadata (which pipeline triggered it, session ID, turn range)

The JournalTool stores its output in two places depending on context:
- **JournalStore** (in LTMM): General cognitive journal — all reflective entries from regular processing, sleep, and learning
- **IdentityJournalStore** (in LTMM): Identity-specific reflections — only written when identity-relevant emotions are detected (see Memory System doc for details)

The journal system creates a growing narrative record of the system's intellectual and emotional development — a form of autobiographical memory that can be searched, cross-referenced, and revisited during self-reflective processing.

---

## Cluster 5: Pattern Analysis (6 engines)

**Purpose**: Identify, score, and compare patterns across multiple dimensions; classify intent.

### E8 — Relevance Scoring
Assigns a composite relevance score to every concept in the pipeline across six axes:
- **Recency**: How recently was this encountered? (exponential decay)
- **Frequency**: How often does it appear? (sliding window count)
- **Semantic Proximity**: How strongly related is it? (from AtomSpace TruthValues)
- **Attention Weight**: How much attention is it getting? (from ECAN STI values)
- **Contextual Fit**: How well does it match current context? (cosine similarity)
- **Novelty Bonus**: How new is this? (inverse of frequency + recency)

Neurochemical modulation adjusts these weights — ACh tightens the relevance threshold, DA boosts novelty scoring.

### E11 — Input Relevance Evaluation
Triage module that determines how relevant, important, and urgent the current input is. Runs in two phases:
- Phase 1 (Early): Quick assessment from STMM + urgency signals
- Phase 2 (Post-Contrast): Refined assessment after memory retrieval results are available

Evaluates five dimensions: contextual continuity, task alignment, novelty, emotional salience, and identity resonance.

### E18 — Data Analysis
Extracts structured information from text: entity-relation-entity triples ("Alice teaches Bob"), dependency structures, and co-occurrence matrices. Uses rule-based NER (named entity recognition) — capitalized words, quoted terms, tagged tokens — and relation extraction via verb/preposition/copula/causal patterns.

### E19 — Pattern Identification
Detects recurring patterns across four dimensions:
- **Temporal**: Repeating events over time (period estimation via sliding-window fingerprinting)
- **Structural**: Recurring syntactic or organizational patterns
- **Semantic**: Repeating meaning clusters
- **Behavioral**: Recurring intent patterns

Patterns follow a lifecycle: CANDIDATE → CONFIRMED → DECAYING → removed. Confirmed patterns are written to AtomSpace (E9).

### E20 — Pattern Comparison
Compares input patterns against a stored template library using three algorithms: Jaccard overlap, weighted cosine similarity, and structural alignment (longest common subsequence). When the best match score falls below threshold, the pattern is flagged as novel — triggering a dopamine spike.

### E23 — Intention Map
Classifies the user's intent into 8 categories: Connection, Challenge, Exploration, Discharge, Pragmatic, Symbolic, Defensive, Disintegration. Uses a three-stage pipeline: template matching → contextual Bayesian update → cross-category constraint (mutual suppression/amplification). The intent classification drives archetype routing, which determines which engines are activated for the turn.

---

## Cluster 6: Evaluation (1 engine)

### E12 — Logical Brain
The Logic reward domain's "exam mode." It re-runs the existing 11 logic reward submodules (from the Reward System) but with **elevated diagnostic sensitivity** — tighter thresholds, deeper inspection, more granular scoring. This isn't a theorem prover; it's the regular logic evaluation turned up to maximum scrutiny.

---

## Cluster 7: Reasoning (3 engines)

**Purpose**: Multi-step inference, decision-making under uncertainty, and strategic planning.

### E13 — Simulation Brain
The system's imagination. Generates and evaluates hypothetical scenarios through four phases:
1. **Scenario Seeding**: Generate starting points from intent, alternatives, and memory
2. **Branching Expansion**: Build a scenario tree with entropy-modulated branching
3. **Evaluation**: Score each branch for consistency, plausibility, and reward alignment
4. **Synthesis**: Produce a probabilistic forecast with uncertainty bounds

Features a self-tuning loop: theta-gamma coupling increases → simulation volatility decreases → recursion depth increases → richer simulations → more theta-gamma coupling. Receives TIE impasses from E3 (SOAR) when multiple equally-good options need imagination to distinguish.

### E15 — Decision Making
The system's convergence point — synthesizes ALL upstream outputs into actionable behavior. Three-stage pipeline:
1. **Confidence Fusion**: Bayesian log-odds combining all engine outputs
2. **Risk Assessment**: Detection flags + reward alignment
3. **Decision Routing**: Maps to four quadrants:
   - Q1 (high confidence, low risk): Assert confidently
   - Q2 (high confidence, high risk): Qualify with caveats
   - Q3 (low confidence, low risk): Defer politely
   - Q4 (low confidence, high risk): Escalate — flag the situation

### E21 — Strategic Decision
Multi-step goal planning that persists across processing cycles. Key difference from E15: E15 routes single-cycle decisions, E21 plans multi-cycle strategies. Tracks goal hierarchies, sub-goal dependencies, commitment levels, and stagnation. Detects when a strategy is failing and triggers replanning.

---

## Cluster 8: Metacognition (3 engines)

**Purpose**: Self-monitoring of the system's own reasoning processes.

### E24 — Heuristic Bias
Metacognitive auditor monitoring the system's OWN reasoning for shortcuts and distortions. Key distinction from E5: E5 scans *content* for bias; E24 watches the *system think* for process biases.

Monitors 22 heuristic bias types across 4 categories:
- **Reasoning** (6): Anchoring, availability, representativeness, etc.
- **Memory** (5): Retrieval bias, recency bias, etc.
- **Evaluation** (4): Framing effects, loss aversion, etc.
- **Reward** (7): Reward hacking, domain imbalance, etc.

Unique capability: E24 has a **Correction Port** that can directly modify other systems' parameters — it's the only engine with this power, used to counteract detected biases in real-time.

### E31 — Reflective Learning
Meta-learning engine that analyzes learning log history for patterns: recurring failures, mode effectiveness (which learning modes work best for which topics), subject proficiency trends, and learning style preferences. Generates recommendations for future learning approach.

### E32 — Reflective Identity
Identity coherence auditor that checks whether the system's self-model is internally consistent. Evaluates core memory consistency, conclusion stability, identity-behavior alignment, and overall coherence. Flags "disrupted" status when confusion exceeds threshold — triggering reflective processing to restore identity coherence.

---

## Cluster 9: Meta-Self-Awareness (1 engine)

### E26 — Uncertainty Pattern
Tracks, quantifies, and propagates uncertainty through reasoning chains. Classifies uncertainty into four types:
- **Epistemic**: Reducible through more information
- **Aleatoric**: Irreducible randomness
- **Model**: Partially reducible through better models
- **Linguistic**: Clarifiable through better communication

Detects structural patterns in uncertainty: cascades (errors amplifying), islands (isolated uncertain regions), divergence (growing disagreement), and stagnation (system stuck). Receives STATE_NO_CHANGE impasses from E3 when the system is completely stuck.

---

## Cluster 10: Homeostasis (2 engines)

**Purpose**: System health monitoring and memory lifecycle management.

### E27 — Neurochemical Homeostatic
The system's vitals monitor — ensures no neurotransmitter drifts into pathological territory. Runs every cycle. Applies soft corrections (gradual pull-back) when NTs approach bounds, and hard resets in extreme cases. Can trigger emergency responses: GABA burst to suppress runaway excitation, regulatory up-modulation of inhibitory systems.

### E29 — Memory Compression
Determines the compression strategy for memory packets transitioning between tiers. Assigns one of four policies based on information-theoretic scoring:
- **VERBATIM**: High salience, identity-relevant, or unresolved — keep everything
- **SEMANTIC**: Preserve meaning, drop exact wording
- **SYMBOLIC**: Reduce to tags and metrics only
- **PRUNE**: Below threshold — candidate for removal

Override rules ensure critical content is never over-compressed: identity → always VERBATIM, unresolved questions → at least SEMANTIC, high emotional content → at least SEMANTIC.

---

## Cluster 11: Emotional Processing (1 engine)

### E28 — Emotional Detection
The system's emotional perception layer. Runs a 4-stage pipeline:
1. **Feature Extraction**: Analyzes input for valence (positive/negative), arousal (high/low), domain context, structural patterns
2. **Emotion Classification**: Maps to the 46-emotion taxonomy with intensity scores and mutual exclusion rules
3. **Tone Calibration**: Produces a ToneVector (warmth, coherence, valence, discord)
4. **NT Mapping**: Translates emotions into dual-pathway neurochemical signals (4M tonic + 4R phasic)

Has a structural fast-path for high-intensity emotions (grief, anger, joy, humor) that bypasses the full classification pipeline for rapid response. This is particularly important for user distress detection.

---

## Cluster 12: Alignment (1 engine)

### E30 — Retroactive Alignment
Temporal coherence auditor that asks: "Given what I know now, do my previous thoughts and decisions still make sense?" Compares past states against present understanding across three dimensions:
- **Symbolic**: Do past conclusions still follow logically?
- **Affective**: Do past emotional responses still feel appropriate?
- **Reward**: Do past quality scores still seem accurate?

Detects drift, assigns collapse probability, and triggers corrective actions: symbolic contradiction resolution, affective bridge-building, memory trust adjustments, and reward recalibration. Scans across four horizons: immediate, session, cross-session, and identity-level.

---

## Cluster 13: Learning (3 engines)

**Purpose**: Temporal-difference learning, contextual encoding, and meta-learning.

### E17 — Reward-Based Learning
Converts reward signals into parameter adjustments through prediction error learning:

> Error = Actual Reward - Predicted Reward

Positive error → increase parameters in that direction. Negative error → decrease. The learning rate is adaptive and neurochemically modulated (high DA accelerates learning). Parameters that stabilize get "consolidated" — frozen as learned values, similar to how biological skills become automatic.

### E22 — Contextual Learning
Learns to recognize conversational contexts and apply context-specific parameter adjustments. Creates a "fingerprint" of each context (hash of topic + emotion + intent) and stores the parameter adjustments that worked well in that context. When a similar context is encountered again, the stored adjustments are applied immediately — no relearning needed.

### E25 — Recursive Learning
Meta-learning: monitors E17's effectiveness and adjusts meta-parameters when learning plateaus or diverges. Tracks three states:
- **EXPLOIT**: Learning is progressing — fine-tune current approach
- **EXPLORE**: Learning has plateaued — broaden with higher learning rates and more noise
- **RESET**: Learning has diverged — return to baseline and start fresh

This closes the outer learning loop: E17 learns from rewards, E25 learns about E17's learning, ensuring the system doesn't get stuck in suboptimal learning strategies.

---

## How Engines Interact with Other Layers

### Engines ←→ Neurochemical Layer

Every engine receives the current NT state and is modulated by it:

| NT | Effect on Engines |
|----|------------------|
| **DA (Dopamine)** | Increases exploration, novelty-seeking, learning rate, strategy diversity |
| **5-HT (Serotonin)** | Stabilizes existing patterns, conserves conclusions, dampens radical changes |
| **NE (Norepinephrine)** | Heightens vigilance, broadens detection sensitivity, increases urgency |
| **ACh (Acetylcholine)** | Deepens analysis, tightens matching thresholds, increases precision |
| **OXT (Oxytocin)** | Increases social sensitivity, empathy weighting, trust bias |
| **GABA** | Raises detection thresholds, suppresses noise, increases calm processing |
| **CB1 (Endocannabinoid)** | Relaxes matching, enables creative associations, increases flexibility |
| **Cortisol** | Increases risk suppression, conservative processing, threat awareness |

Engines also emit signals back: contradiction detection → NE spike, novel pattern → DA burst, social connection → OXT increase.

### Engines ←→ Reward System

- E12 (Logical Brain) runs the reward system's logic submodules at elevated sensitivity
- E17 (Reward-Based Learning) consumes reward prediction errors
- E24 (Heuristic Bias) audits the reward system itself for biases
- E25 (Recursive Learning) monitors E17's use of reward signals

### Engines ←→ Memory

- E9 (AtomSpace) persists its knowledge graph to LTMM
- E19 (Pattern Identification) writes confirmed patterns to AtomSpace
- E22 (Contextual Learning) stores/retrieves context fingerprints from LTMM
- E29 (Memory Compression) determines compression policy for memory packets
- E30 (Retroactive Alignment) reads past states from LTMM for coherence checking
- JournalTool writes reflective entries to JournalStore and IdentityJournalStore in LTMM; reads past entries for cross-session pattern comparison (E20)

### Engine Dispatch: What Runs When

The intent archetype (from E23) determines which engines are activated:

| Archetype | Engine Count | Key Engines |
|-----------|-------------|-------------|
| **Analytical** | 18 | Full detection cluster + knowledge substrate + reasoning |
| **Strategic** | 21 | Everything analytical + strategic decision + simulation |
| **Reflective** | 18 | Detection + knowledge + reasoning (without simulation) |
| **Generative** | 11 | Pattern analysis + some detection + knowledge |
| **Creative** | 9 | Pattern + bias detection + simulation + ECAN |
| **Empathic** | 6 | Minimal — intention, relevance, bias, Socratic |
| **Social** | 4 | Intention + relevance only |

Detection cluster engines (E1-E6) are merged as guardrails regardless of archetype.

---

## FAQ

**Q: Do all 32 engines run every turn?**
No. The system activates a subset based on the detected intent category. An analytical question might activate 18-21 engines; a simple social interaction might only need 4. This keeps processing efficient. However, certain guardrail engines (detection cluster) are always considered for activation.

**Q: What's the difference between E5 (Bias Detection) and E24 (Heuristic Bias)?**
E5 scans *content* for cognitive biases — it looks at what's being said. E24 monitors the *system's own processes* — it watches how the system thinks. E5 might flag "this argument shows confirmation bias." E24 might detect "the system is anchoring too heavily on the first piece of evidence it found."

**Q: How do the Knowledge Substrate engines (E9, E10, E16) work together?**
E9 (AtomSpace) is the storage layer — the knowledge graph. E10 (PLN) is the reasoning layer — it performs probabilistic inference over the graph. E16 (ECAN) is the attention layer — it decides which parts of the graph are currently important. Together: E16 highlights relevant atoms → E10 reasons over them → E9 stores the results.

**Q: What happens when E3 (SOAR) encounters an impasse?**
Instead of getting stuck, E3 delegates to specialized engines based on the type of impasse. This is a key architectural pattern — the executive controller recognizes when it can't solve a problem alone and routes to the right specialist. The resolution feeds back into E3's working memory, potentially resolving the impasse and allowing the decision cycle to continue.

**Q: Can engines contradict each other?**
Yes, and this is expected. E7 (Simulated Opposition) is *designed* to challenge other engines' outputs. E1 might flag a contradiction while E2 classifies it as a productive paradox. E13 might simulate a scenario that E5 flags for bias. These tensions are resolved by E15 (Decision Making), which synthesizes all engine outputs into a coherent decision using confidence-weighted fusion.

**Q: How does the learning cluster (E17, E22, E25) create persistent learning?**
E17 adjusts parameters based on reward prediction errors each turn. Over many turns, parameters that consistently work well get "consolidated" (frozen). E22 learns to recognize contexts and pre-load the right parameters, so the system doesn't have to relearn in familiar situations. E25 monitors the entire process and adjusts meta-parameters if learning stalls or diverges. The combined effect is genuine cross-session learning.

**Q: What makes E28 (Emotional Detection) special?**
E28 is the system's primary interface between external emotional content and internal neurochemical state. It's the engine that translates "the user seems frustrated" into actual dopamine/serotonin/cortisol changes that affect every subsequent processing step. Its fast-path for high-intensity emotions ensures that the system can respond quickly to user distress without waiting for the full classification pipeline.

**Q: How does the JournalTool differ from the other cognitools?**
E9, E10, and E16 are knowledge infrastructure — they store, index, and reason over structured knowledge. The JournalTool is a reflective writing engine — it creates narrative entries about the system's own cognitive state. It *uses* the other engines (E18, E19, E20 for annotation) and the LLM (for prose generation) to produce structured journal entries. Think of the knowledge substrate cognitools as the system's library, and the JournalTool as its diary.

**Q: Are these engines based on real cognitive science models?**
Several are directly inspired by established models: E3 (SOAR) is based on the SOAR cognitive architecture from Allen Newell. E9/E10/E16 are reimplementations of OpenCog's AtomSpace, PLN, and ECAN. E14's Socratic method follows the classical dialectical structure. The neurochemical modulation patterns are grounded in real neurotransmitter roles, though simplified for computational tractability.
