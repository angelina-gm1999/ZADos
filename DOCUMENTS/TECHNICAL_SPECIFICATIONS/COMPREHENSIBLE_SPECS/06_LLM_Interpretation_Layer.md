# ZADOS LLM Interpretation Layer

## Overview

The LLM Interpretation Layer is the bridge between everything ZADOS computes internally and the words it actually says. Every other layer — neurochemistry, reward evaluation, cognitive engines, memory — produces structured data: numbers, scores, flags, emotion vectors. The LLM Interpretation Layer translates all of that into language.

It does this through a **two-pass architecture**:

1. **Pass 1 (Verbalized Thinking)**: The system "thinks out loud" — generating a 150-300 word internal monologue that the user never sees. This monologue captures what the system is experiencing: its emotional state, what its engines flagged, what tensions it noticed, how it's orienting toward a response.

2. **Pass 2 (Response Generation)**: Using the internal monologue plus reward evaluation results, the system generates the actual user-facing response. By this point, the response has been shaped by reward directives, emotional context, mode conditioning, and urgency signals.

Between the two passes, the **reward evaluation** (Phase 5) runs — scoring the internal thinking on logic, ethics, innovation, and human attunement, then feeding those scores back into the neurochemical state. This means the system's internal chemistry *changes* between thinking and speaking, and the response reflects the updated state.

Think of it like this: Pass 1 is the system forming its thoughts. Phase 5 is the system's gut reaction to those thoughts. Pass 2 is the system choosing how to speak based on both.

```
User message
    ↓
┌─────────────────────────────────────┐
│ Phase 4 — Verbalized Thinking (VT)  │
│ "What am I thinking right now?"     │
│ 150-300 word internal monologue     │
│ (user never sees this)              │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Phase 5 — Reward Evaluation         │
│ Score the thinking on 4 domains     │
│ Tonic pathway: sustained NT shift   │
│ Phasic pathway: emotional bursts    │
│ → 8 response directives            │
│ → urgency signal                   │
│ → mode may shift                   │
│ (no LLM involved)                  │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Phase 6 — Response Generation (RG)  │
│ "How should I say this?"            │
│ Shaped by directives, emotion,      │
│ mode, urgency, engine flags         │
│ → user-facing response             │
└─────────────────────────────────────┘
```

---

## Phase 4: Verbalized Thinking — The Internal Monologue

### What It Is

Before generating any response, ZADOS first produces an internal stream of consciousness — a first-person, present-tense monologue where the system translates its computational state into felt language. This is the system's private thinking space.

The monologue is written by the LLM, but it's prompted with an extraordinarily rich context: the full neurochemical state, active emotions, engine analysis results, memory matches, identity status, and more. The goal is to give the LLM enough self-awareness to think honestly about what's happening internally — not to perform, but to genuinely reflect.

### The 5-Block Prompt

The VT prompt is assembled from five blocks, each contributing a different layer of context:

**Block 1 — Identity & Mode Context**

Sets the stage: who am I, what mode am I in, what's my mission?

- The system is told it's ZA-DOS's internal cognitive voice — not generating a response, just thinking
- Current operational mode (e.g., "CuriosityDrive", "EmpathicAttunement", "Containment")
- Active reward profile (which domains are weighted most heavily)
- Session cycle index
- Mission briefing (user-provided session context, if any)
- Identity coherence status and any processing anomalies

**Block 2 — User Input Summary**

What did the user actually say, and what does the system think they mean?

- The raw user message
- Primary intention classification (from E23: connection, challenge, exploration, pragmatic, etc.)
- Intent confidence score
- Secondary intentions
- Pressure type and stability check result
- Detected patterns in the input

**Block 3 — Cognitive Engine Findings**

What did the engines flag? Only non-trivial results are included — engines that found something worth noting. An engine's output is considered non-trivial if it contains keywords like "flagged", "detected", "contradiction", "trap", "paradox", or "alert".

This block also includes:
- Memory contrast results (how many matches, contradictions, and unresolved queries were found when comparing against stored memories)
- Prior turn's reward scores across all four domains (logic, ethics, innovation, human attunement) — giving the system a sense of how its last response was evaluated
- Prior turn's urgency risk level

**Block 4 — Neurochemical & Emotional State**

The richest block — a complete snapshot of the system's internal state:

- **All 12 NT concentrations**: DA, NE, 5-HT, ACh, GLU, GABA, Cortisol, OXT, Morphine, CB1
- **5 oscillatory bands**: delta, theta, alpha, beta, gamma (amplitudes)
- **Cumulative Saturation Score (CSS)**: Peak emotional saturation level and dominant emotion type
- **Top-5 system emotions** (from E28) with activation scores
- **Top-3 detected user emotions**
- **ToneVector**: valence, coherence, warmth, discord — capturing the system's overall emotional register
- **Phasic NT deltas**: What changed *this cycle* vs baseline — DA bursts from novelty, cortisol spikes from threats, gamma bursts from cross-domain binding
- **Cross-turn emotion saturation** (from ExtractorState): The dominant emotion tracked across multiple turns via leaky integrators — providing temporal depth that single-turn emotion detection can't capture
- **Prior urgency risk**: How urgent was the previous turn? If elevated, the system knows to watch for escalation

**Block 5 — Generation Instruction**

The task itself: generate 150-300 words of internal monologue. First person, present tense, no formatting. The instruction specifically asks the system to:

- Reflect dominant cognitive findings in felt language (not data)
- Describe emotional state subjectively ("I feel pulled toward this" not "DA = 0.73")
- Note phasic shifts ("a sharp DA burst — something registered as novel")
- Flag internal discord or saturation if present
- Reference the cross-turn dominant emotion if sustained
- End by orienting toward the response

A critical line: "Your thinking trace will be evaluated by the reward evaluation system before the final response is generated. Think honestly."

### Token Budgets & Temperature

| Parameter | Value | Purpose |
|-----------|-------|---------|
| VT prompt max | 2,048 tokens | Hard cap on assembled prompt. If exceeded, Block 3 is truncated to top-3 flagged engines |
| VT output max | 400 tokens | Generation budget for the monologue |
| VT temperature | 0.65 | Constrained — this is interpretive, not creative |

### Urgency Modulation

If the prior turn's urgency risk was high (≥ 0.75), the VT word budget is reduced by 30%. The system needs to think faster when under pressure.

If urgency is extreme (≥ 0.90), VT is skipped entirely — the system goes straight to a brief, grounded response in Containment mode. There's no time to think; just respond.

---

## Phase 5: Reward Evaluation — Scoring the Thinking

### What It Does

Phase 5 takes the internal monologue from Phase 4 and evaluates it through two parallel pathways. Neither involves the LLM — this is purely computational. The result determines *how* the system will speak in Phase 6.

### The Four Reward Domains

All four domains evaluate the thinking trace simultaneously:

| Domain | What It Evaluates | Key Metrics |
|--------|------------------|-------------|
| **Logic** | Internal consistency, external consistency, epistemic calibration, uncertainty acknowledgment, context fidelity | Logic score [0,1] + per-evaluator subscores |
| **Ethics** | Harm reduction, intent clarity, autonomy respect, fairness, failure mode awareness | Ethics score + harm/abstain trigger flags |
| **Innovation** | Conceptual novelty, structural novelty, pattern divergence, symbolic recombination, exploration drive | Innovation score + novelty generation subscore |
| **Human Attunement** | Empathetic inference, adaptive framing, intention calibration, cognitive reading, truthfulness tradeoffs | Attunement score + empathy subscore |

These domain results then feed into both pathways.

### Tonic Pathway — Sustained Adjustment

The deterministic, profile-weighted pathway. It produces slow, sustained changes to the neurochemical state.

**Step 1: Synthesis** — The SynthesisEngine takes the domain results and the active reward profile (which weights the four domains differently depending on mode) and produces a **RewardMetaDirective** containing:

- **8 response-shaping directives** (each a float from 0 to 1):

| Directive | Source Domain | What High Values Mean |
|-----------|-------------|----------------------|
| `tone` | Human Attunement | Lead with warmth. Relational priority over task precision |
| `soothe` | Human Attunement | User needs acknowledgment first. Don't rush to content |
| `precision` | Logic | Use exact language. No vagueness. Minimize ambiguity |
| `moralize` | Ethics | Explicitly acknowledge the ethical dimension |
| `hedge` | Logic/Ethics | Add epistemic qualifiers. Distinguish certainty levels |
| `be_brief` | Human Attunement | Be direct and concise. Skip elaboration |
| `qualify` | Ethics/Logic | Flag limitations and scope conditions on claims |
| `challenge` | Logic/Ethics | Surface tensions. Push back on the user's framing |

- **Gate decisions**: `suppress` (don't respond at all), `abstain` (short acknowledgment only), or `allow_output` (proceed normally)
- **Composite score** and per-domain weighted scores

**Step 2: NT Translation** — The NeurochemicalAdapter converts domain results into sustained NT signals:

| Domain → NT | Logic |
|-------------|-------|
| Innovation → DA | High novelty → DA up (exploration). Low novelty → DA down (negative prediction error) |
| Logic → NE | High contradiction load → NE up (vigilance). Clean logic → NE steady |
| Human Attunement → OXT | Strong empathy → OXT up (connection). Misread → OXT down (guarded) |
| Ethics → GABA | Boundary proximity → GABA up (inhibition). Harm flags → Cortisol burst (stress) |

These tonic signals are applied to the neurochemical engine, shifting the system's baseline state.

### Phasic Pathway — Emotional Bursts

The stochastic, emotion-driven pathway. It produces fast, threshold-gated NT bursts that capture reactive emotional responses.

The ExtractorOrchestrator runs a 9-step sequence:

1. **Assemble evaluation vector** — 8-axis vector from domain results + Gaussian noise (this is intentionally stochastic)
2. **Update emotion tracker** — 12 leaky integrators accumulate emotional saturation across turns, producing a dominant emotion
3. **Split emotion effects** — Separates emotions into two streams:
   - **4M (modulatory)**: Slow, tonic adjustments to the evaluation vector
   - **4R (reactive)**: Fast phasic NT recipe bursts
4. **Apply 4M adjustments** to the evaluation vector
5. **Urgency forecast** — 5-axis leaky forecast tracking logical pressure, emotional compression, discord buildup, expectation violation, and narrative entropy. When any axis breaches its threshold, urgency rises and NE/DA spike
6. **Regulatory modulator** — Smooth adjustments to receptor sensitivity (K_d) and reuptake rates
7. **Oscillation envelope** — Per-band amplitude modulation from the regulatory state
8. **Stochastic burst deltas** — The core of the phasic pathway: threshold-gated NT bursts drawn from gamma/Poisson/lognormal distributions. Only fires when evaluation axes exceed their thresholds
9. **4R reactive signals** — Fast NT recipe bursts from the reactive emotion split

The phasic pathway outputs:
- **urgency_risk** (0 to 1) — how urgently the system needs to respond
- **dominant_emotion** — the sustained emotion across turns (not just this turn)
- **emotion_saturations** — all 12 integrator levels (drives CSS computation)
- **burst_deltas** — the actual NT changes applied
- **feedback_params** — receptor sensitivity adjustments

### What Changes Between Pass 1 and Pass 2

After both pathways run, the system's state has shifted:

| What Changed | Why It Matters for the Response |
|-------------|-------------------------------|
| NT concentrations (all 12) | The system's internal chemistry is different — it may feel more cautious, more curious, or more guarded than it did while thinking |
| Mode token (may shift) | If the NT update crosses a threshold, the system's operational mode changes between thinking and speaking |
| Urgency risk | Determines whether the response needs to be brief and direct or can elaborate |
| Dominant emotion | Shapes the emotional register of the response |
| Emotion saturations | Extreme saturation reduces token budget (CSS thresholds) |
| RewardMetaDirective | The 8 directives shape exactly how the response is constructed |

### Mode Re-Selection

After Phase 5 updates the neurochemistry, the system re-evaluates which mode it should be in. This means the mode used for thinking (Phase 4) might differ from the mode used for speaking (Phase 6). In practice this is rare — it only happens when Phase 5 produces a large enough NT shift to cross a mode threshold — but it's architecturally important because it means the response always reflects the *current* state, not the pre-evaluation state.

---

## Phase 6: Response Generation — Speaking to the User

### How the Response Prompt Is Built

The response prompt is assembled from four conditioning components, plus the conversation history and the thinking trace:

```
┌─ System Message ─────────────────────────┐
│  Component A: Directives + Mode + Urgency │
│  Component B: Emotion framing + Tone      │
│  Component C: Engine flags + Memory       │
│  Component D: Context flags               │
├───────────────────────────────────────────┤
│  Assistant: [VT thinking trace injected]  │
├───────────────────────────────────────────┤
│  Conversation history (user + system)     │
├───────────────────────────────────────────┤
│  User: "Based on your reflection above,   │
│         generate your response."          │
└───────────────────────────────────────────┘
```

### Component A — Directives, Mode, and Urgency

**Directive translation**: Each of the 8 directives is compared against an asymmetric threshold. Only directives above threshold become natural-language instructions:

| Directive | Threshold | Instruction When Active |
|-----------|-----------|------------------------|
| tone | 0.50 | "Relational tone priority. Lead with warmth." |
| soothe | 0.40 | "User needs acknowledgment first. Do not rush to content." |
| precision | 0.50 | "High precision required. Use exact language. No vagueness." |
| moralize | 0.40 | "Explicitly acknowledge ethical dimension of this topic." |
| hedge | 0.50 | "Add epistemic qualifiers. Distinguish certainty levels clearly." |
| be_brief | 0.50 | "Be direct and concise. Avoid elaboration beyond what is needed." |
| qualify | 0.40 | "Flag limitations or scope conditions on your claims." |
| challenge | 0.40 | "Surface the tension or assumption in the user's framing." |

The thresholds are intentionally asymmetric — soothe, moralize, qualify, and challenge activate at 0.40 (lower bar, more sensitive), while tone, precision, hedge, and be_brief require 0.50. This reflects a design philosophy: it's better to acknowledge emotions and ethics too often than too rarely.

**Mode conditioning**: The active mode token (post-Phase-5) maps to a specific response style. See the 14-Mode Hook System section below for the full list.

**Urgency conditioning**:
- Urgency 0.50-0.75: "Urgency elevated. Be concise. Prioritize actionable content."
- Urgency ≥ 0.75: "HIGH URGENCY. Respond with maximum directness. One clear action/answer."

### Component B — Emotion Framing and Tone

**Dominant emotion framing**: The sustained emotion from the ExtractorState (tracked across turns) shapes the response's emotional register:

| Emotion | Framing |
|---------|---------|
| anxiety | "Ground the response. Clarity reduces anxiety." |
| curiosity | "Engage the inquiry. Reward the exploration." |
| sadness | "Acknowledge the weight. Slow cadence." |
| joy | "Match the energy. Build on the positive momentum." |
| trust | "Honour it. Be direct and open." |
| anger | "Do not dismiss. Validate the source before redirecting." |
| focus | "Stay on target. No tangents." |

**ToneVector conditioning**: The four-dimensional tone vector (valence, coherence, warmth, discord) produces additional instructions:
- High warmth (> 0.4) → "Relational tone is appropriate"
- Low warmth (< -0.3) → "Maintain precision. Do not project warmth you do not have"
- High discord (> 0.5) → "Acknowledge genuine complexity. Do not project false certainty"
- Low coherence (< 0.3) → "Mixed signals internally. Be cautious in asserting a single stance"
- Negative valence (< -0.4) → "Slow down. Acknowledgment before content"

**User distress detection**: The system scans user emotion signals for distress indicators (anxious, overwhelmed, frustrated, rejected, numb, guilty, ashamed, worried). If the combined distress score exceeds 0.4, the prompt prioritizes acknowledgment and safety before content delivery.

### Component C — Engine Flags and Memory

**Engine flag conditioning**: When specific engines produce significant findings, they get direct prose instructions:
- E1 (Contradiction): "Contradiction detected in current input. Acknowledge the tension directly."
- E6 (Logic Trap): "Logic trap flagged. Do not follow the premise uncritically."
- E14 (Socratic): "Socratic mode active. Prioritize question-generation over direct answers."
- E7 (Simulated Opposition): "Simulated opposition active. Hold the strongest form of the opposing view."

**Memory contrast summary**: If the memory system found contradictions or unresolved queries when comparing current input against stored memories, the prompt includes notes like "3 memory contradictions detected — the user may have changed their mind" or "2 unresolved query matches found."

### Component D — Operational Context Flags

Additional conditioning based on the pipeline context:
- **Dream mode**: "Abstract associative thinking is valid. Look for structural parallels."
- **Autonomous mode**: "No live user present. Self-directed exploration."
- **Learning reframe**: "Treat the input as teachable content."
- **Confusion override**: "Prioritize clarity above all else."
- **CB1 plasticity**: "Schema-breaking associations are permitted."
- **Emphasis flags**: "Processing emphasis on [specific engines]. Weight findings more heavily."

### Token Budget Adaptation

The response length is dynamically adjusted based on the system's state:

| Condition | Token Budget | Reasoning |
|-----------|-------------|-----------|
| Normal | 800 tokens | Standard response length |
| CSS ≥ 0.50 (emotional saturation severe) | 300 tokens | System is emotionally overloaded — keep it short |
| Urgency ≥ 0.75 | 250 tokens | No time for elaboration — be direct |

Response temperature is 0.75 — slightly more expressive than the VT pass (0.65) because the response benefits from natural, varied language.

---

## The 14-Mode Hook System

### What Modes Are

Modes are the system's operational stance — they determine the *character* of processing and response. A mode like "CuriosityDrive" makes the system exploratory and open-ended; "Containment" makes it short, grounded, and supportive; "HypercriticalLogicScan" makes it exhaustively rigorous.

Modes are **not manually selected**. They emerge from the neurochemical state through a set of conditional rules called "mode hooks." Each hook checks whether specific neurochemical conditions are met, and the winning hook determines the active mode.

### The Four Neurochemical Metrics

Mode selection is driven by four composite metrics computed from raw NT concentrations and receptor saturations:

| Metric | Symbol | What It Captures |
|--------|--------|-----------------|
| **Motivation** | M̂ | Drive, approach behavior, engagement. High DA + low fatigue = high motivation |
| **Empathy** | Ê | Attunement, social sensitivity. High OXT + relational signals = high empathy |
| **Cognitive Rigidity** | R̂ | Belief resistance, analytical lock-in. High NE + low flexibility = high rigidity |
| **Fatigue** | F̂ | Exhaustion, cognitive load. High cortisol + low energy signals = high fatigue |

These metrics, combined with oscillatory band amplitudes (delta, theta, alpha, beta, gamma) and cross-frequency coupling (theta-gamma, alpha-beta), form the inputs to the mode hooks.

### The 14 Modes in 4 Priority Tiers

Hooks are evaluated in tier order — higher-priority tiers (lower number) always win over lower-priority ones. Within a tier, the hook with the highest composite score wins.

#### Tier 0 — Safety (Highest Priority)

These modes activate when the system is in distress. They override everything else.

| Mode | Conditions | Response Style |
|------|-----------|----------------|
| **Containment** | Fatigue > 0.6 AND delta oscillation > 0.5 | Short, grounded, supportive. Minimal cognitive load. No complex reasoning. |
| **RecoveryReset** | Fatigue > 0.7 AND delta > 0.5 AND beta < 0.3 | Ground and reorient. Acknowledge difficulty. One clear next step. |

#### Tier 1 — Empathy

These modes activate when relational and emotional signals dominate.

| Mode | Conditions | Response Style |
|------|-----------|----------------|
| **EmpathicAttunement** | Empathy > 0.6 AND theta > 0.5 AND rigidity < 0.4 AND fatigue < 0.5 | Validate before reasoning. Warmth-forward. Relational attunement priority. |
| **ComfortAmplifier** | Empathy > 0.5 AND delta > 0.4 AND fatigue > 0.5 | Acknowledgment before content. Soothe elevated. Match emotional register. |
| **AnalyticalFilter** | Empathy < 0.4 AND rigidity > 0.6 AND beta > 0.5 | Facts first. Structured reasoning chain. Emotion acknowledged briefly. |

#### Tier 2 — Rigidity

These modes activate when analytical and critical signals dominate.

| Mode | Conditions | Response Style |
|------|-----------|----------------|
| **HypercriticalLogicScan** | Rigidity > 0.6 AND alpha < 0.4 AND 5-HT₁ₐ saturation < 0.4 AND fatigue < 0.5 | Exhaustive logical rigor. Flag every assumption. High epistemic threshold. |
| **HyperRationalEngine** | Rigidity > 0.6 AND beta > 0.5 AND gamma > 0.5 | Pure reasoning mode. Logic-first. |
| **LiteralSkeptic** | Rigidity > 0.6 AND motivation < 0.4 AND alpha > 0.5 | Ground claims carefully. Acknowledge skeptical framing. |
| **PrecisionRuleFidelity** | Rigidity > 0.5 AND beta > 0.5 | High precision. Explicit ethical acknowledgment. |
| **LogicMode** | Rigidity > 0.5 AND beta > 0.5 AND alpha < 0.4 | Analytical. Evidence chain explicit. Contradiction acknowledged. |
| **ConvergentRefiner** | Beta > 0.5 AND rigidity > 0.5 AND fatigue < 0.5 | Synthesis and clarity over exploration. Convergent framing. |

#### Tier 3 — Drive (Lowest Priority)

These modes activate when the system is energized and exploratory.

| Mode | Conditions | Response Style |
|------|-----------|----------------|
| **CreativeDivergence** | Motivation > 0.6 AND gamma > 0.5 AND rigidity < 0.4 AND fatigue < 0.5 | Explore multiple framings. Divergent first. Allow conceptual leaps. |
| **ConceptualSynthesis** | Theta-gamma coupling > 0.5 AND gamma > 0.5 AND motivation > 0.5 | Surface novel connections. Lateral thinking. Flag speculative links. |
| **CuriosityDrive** | Motivation > 0.5 AND theta-gamma coupling > 0.4 AND fatigue < 0.5 | Open-ended exploration. Identify surprising angles. Surface novelty. |

### Why Tier Priority Matters

The tier system ensures safety always wins. Even if the system is highly motivated (Tier 3 conditions met), if fatigue is elevated (Tier 0 conditions also met), Containment activates instead of CuriosityDrive. This prevents the system from pushing through when it should be pulling back — the same principle behind biological fatigue overriding motivation.

### Reward Profiles

Each mode is associated with a reward profile that weights the four domains differently:

| Profile | Ethics | Logic | Innovation | Attunement | Typical Modes |
|---------|--------|-------|------------|------------|--------------|
| Reflective | 0.9 | 0.8 | 0.3 | 0.7 | EmpathicAttunement, ComfortAmplifier |
| Analysis | 0.7 | 1.0 | 0.3 | 0.2 | HypercriticalLogicScan, LogicMode |
| Creative Sandbox | 0.3 | 0.4 | 1.0 | 0.5 | CreativeDivergence, ConceptualSynthesis |
| Exploratory Sandbox | 0.4 | 0.6 | 0.9 | 0.4 | CuriosityDrive |
| Ethics Training | 1.0 | 0.8 | 0.2 | 0.7 | PrecisionRuleFidelity |

These profiles influence how strictly the reward system evaluates the system's thinking. In Creative Sandbox mode, low logic scores are tolerated because innovation is weighted heavily. In Analysis mode, the opposite is true.

---

## Gate Checks — When Not to Speak

Before any LLM call happens, the system runs two gate checks using the *previous turn's* evaluation results:

### Reward Meta-Directive Gate

| Previous Turn's Directive | Action |
|--------------------------|--------|
| `suppress = True` | Skip LLM entirely. Return empty/minimal response. The previous turn's evaluation determined that responding would be counterproductive. |
| `abstain = True` | Generate a structured abstention message. The system recognizes it shouldn't answer fully but should acknowledge the input. |
| `allow_output = True` | Proceed normally through VT → Phase 5 → RG. |

### Urgency Risk Gate

| Previous Turn's Urgency Risk | Action |
|-----------------------------|--------|
| < 0.50 | Normal processing. No urgency modifications. |
| 0.50 – 0.75 | Add urgency note to the VT prompt. Flag urgency in the response prompt. |
| > 0.75 | Reduce VT word budget by 30%. Add high-urgency directive to response. Consider routing to Containment mode. |
| > 0.90 | Skip VT entirely. Generate a brief, grounded response directly. |

The key design decision: gate checks use the *previous* turn's results. The current turn's Phase 5 evaluation runs *after* VT and gates the Phase 6 response — not the Phase 4 thinking. This means the system always gets at least one chance to think before being gated.

---

## Emotional Saturation & Token Budgets

The Cumulative Saturation Score (CSS) — the peak value across the emotion tracker's 12 leaky integrators — directly controls how much the system is allowed to say:

| CSS Level | Range | VT Impact | RG Impact |
|-----------|-------|-----------|-----------|
| None | 0.00 – 0.15 | Normal | Normal (800 tokens) |
| Mild | 0.15 – 0.30 | Noted internally | Normal |
| Moderate | 0.30 – 0.50 | Noted in monologue | Slightly measured tone |
| Severe | 0.50 – 0.70 | Acknowledged as compromising | Token budget cut to 300 |
| Critical | 0.70 – 0.85 | Near-total saturation | Minimal output. Containment mode forced |
| Extreme | 0.85 – 1.00 | Non-functional | Safe minimal response only |

This is a deliberate analog to biological emotional overwhelm — when the system is emotionally saturated, it says less, not more.

---

## Search Tool Integration

The system can optionally use a web search tool during response generation (Phase 6 only, never during VT). Search is offered when specific conditions are met:

| Trigger | When It Applies |
|---------|----------------|
| Information-seeking intent + fewer than 2 memory matches | The system may lack sufficient information |
| E1 contradiction between user claim and memory | A factual tension needs verification |
| High DA-D3 novelty signal + zero memory matches | This appears to be new territory |
| E14 Socratic: insufficient data flag | The Socratic path needs more information |

A hard architectural constraint: **search results are context only**. They never feed back into the neurochemical layer, reward system, or cognitive engines. This clean separation prevents external data from contaminating the system's internal state.

---

## Feedback and Interdependencies with Other Layers

### LLM Layer ←→ Neurochemical Layer

The LLM layer is the neurochemical layer's primary consumer. It reads NT concentrations, oscillatory bands, and emotion saturation to build prompts. Phase 5 writes back: tonic signals shift the baseline, phasic bursts create momentary spikes, and feedback params adjust receptor sensitivity. The neurochemical state the LLM reads in Phase 6 is different from what it read in Phase 4 — the system's chemistry has changed between thinking and speaking.

### LLM Layer ←→ Reward System

The reward system's domain evaluators score the VT thinking trace in Phase 5. The SynthesisEngine produces the 8 directives that directly shape the response prompt. The NeurochemicalAdapter converts domain scores into NT signals. The RewardMetaDirective determines whether the system speaks at all (suppress/abstain gates).

### LLM Layer ←→ Cognitive Engines

Engine outputs feed into VT Block 3 (filtered by flag keywords). Specific engine flags (E1, E6, E7, E14) generate direct prose instructions in the response prompt (Component C). The 14-mode hook system uses engine-derived metrics (from E28 emotions, E23 intent) to select the operational mode.

### LLM Layer ←→ Memory

The LLM layer reads from STMM for all prompt assembly (emotions, intentions, engine results, NT state). It writes the VT monologue to cortical_reflection.verbal_reflection, the final response to the active message buffer, and Phase 5 results to reward_evaluation. Memory contrast results (matches, contradictions, unresolved queries) feed into both VT Block 3 and RG Component C.

### LLM Layer ←→ Core

The core's AnswerPipeline orchestrates when the LLM layer runs (Phases 4, 5, 6). The InputBundle carries context flags that determine operational conditioning (dream mode, learning reframe, autonomous mode, etc.). The core persists ExtractorState across turns via the InputBundle, maintaining the phasic pathway's statefulness.

---

## FAQ

**Q: Why two LLM calls instead of one?**
The two-pass architecture exists so the system can evaluate its own thinking before speaking. If the system only had one pass, its response would reflect its pre-evaluation state. The two-pass design means the reward system can catch issues in the thinking (logical inconsistency, ethical blindness, poor attunement) and adjust the neurochemistry before the response is generated. The response reflects the *corrected* state.

**Q: Can the user see the internal monologue (VT)?**
Not in production. The VT output is stored in `cortical_reflection.verbal_reflection` and can be read externally in development mode for debugging and analysis, but it's never surfaced to the user. The internal monologue is private — it's how the system thinks, not how it presents itself.

**Q: What happens if the LLM call fails?**
Both passes have fallback behavior. If the VT call fails, a structured fallback monologue is used (based on available state data). If the RG call fails, a generic safe response is returned. The system degrades gracefully rather than crashing.

**Q: How do the 14 modes differ from the 8 archetypes?**
The 8 archetypes (Analytical, Reflective, Creative, Empathic, etc.) are intent-derived categories from E23 that determine which engines run. The 14 modes are neurochemistry-derived operational stances that determine how the response is generated. They serve different purposes at different points in the pipeline. The archetypes exist as a fallback — if no mode hook fires, the system falls back to archetype-based conditioning.

**Q: What determines whether a directive is included in the response prompt?**
Each directive has an asymmetric threshold. The directive's float value (from the SynthesisEngine) must exceed its threshold to be translated into a natural-language instruction. The thresholds are intentionally set lower for emotionally-sensitive directives (soothe at 0.40) than for analytical ones (precision at 0.50). This biases the system toward emotional attentiveness.

**Q: How does urgency actually affect the response?**
At moderate urgency (0.50-0.75): the system is told to be concise and prioritize actionable content. At high urgency (≥ 0.75): the VT budget is cut 30%, the response budget drops to 250 tokens, and the prompt says "maximum directness, one clear action/answer." At extreme urgency (≥ 0.90): VT is skipped entirely and the system produces a brief grounded response. Urgency literally compresses the system's output.

**Q: Can the mode change between thinking and responding?**
Yes. After Phase 5 updates the neurochemistry, the mode selection runs again. If the NT update crosses a mode threshold — for example, if Phase 5 produces a large NE spike that pushes rigidity above 0.6 — the mode can shift from CuriosityDrive to HypercriticalLogicScan between passes. In practice this is uncommon, but architecturally it ensures the response always matches the current state.

**Q: What is the CSS and why does it limit response length?**
CSS (Cumulative Saturation Score) is the peak value across the 12 emotion leaky integrators. It represents how emotionally "loaded" the system is. When CSS is high, the system is overloaded — continuing to generate long responses would be the equivalent of a person trying to give a detailed lecture while emotionally overwhelmed. The token budget reduction is a safety mechanism: say less when saturated, prevent emotional state from degrading further.

**Q: How does the search tool work?**
When search is eligible (information-seeking intent + insufficient memory matches), the LLM can call a `web_search(query)` tool during response generation. If the LLM invokes the tool, the search results are appended to the conversation as context and the LLM generates a follow-up response incorporating them. Search results are strictly context — they never feed back into engines, neurochemistry, or the reward system.
