# ZADOS Reward System

## Overview

The Reward System is ZADOS's quality control and behavioral guidance layer. It evaluates everything the system produces across four fundamental domains — Logic, Ethics, Innovation, and Human Attunement — and translates those evaluations into concrete instructions for how the system should respond.

Crucially, this is a **non-punitive** system. It doesn't penalize bad outputs; it shapes behavior through positive reward signals. Think of it less like a judge handing down sentences and more like a coach giving real-time feedback: "more warmth here," "tighten the logic there," "this needs more creative thinking."

The reward system sits at the intersection of cognition and neurochemistry. Its evaluations flow in two directions: forward into response-shaping directives that tell the LLM *how* to write its answer, and backward into neurochemical feedback that adjusts the system's internal state for the next processing cycle.

---

## How It Works: The Evaluation Pipeline

Every turn, the reward system runs a four-stage pipeline:

```
Input State (VT thinking trace + context)
         │
         ▼
┌─────────────────────────────────┐
│   DOMAIN EVALUATION (parallel)  │
│  ┌───────┐ ┌───────┐           │
│  │ Logic │ │Ethics │           │
│  │(8 sub)│ │(9 sub)│           │
│  └───────┘ └───────┘           │
│  ┌──────────┐ ┌─────────────┐  │
│  │Innovation│ │Human Attune.│  │
│  │ (10 sub) │ │  (10 sub)   │  │
│  └──────────┘ └─────────────┘  │
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│      SYNTHESIS ENGINE           │
│  Tier classification            │
│  Weighted composite score       │
│  8 response directives          │
│  Suppress / Abstain decisions   │
│  Routing hints for the LLM      │
└────────────┬────────────────────┘
             ▼
    ┌────────┴────────┐
    ▼                 ▼
┌────────┐    ┌──────────────────┐
│Feedback│    │Neurochemical     │
│Engine  │    │Adapter           │
│(deltas)│    │(NT signal mapping│
└────┬───┘    └────────┬─────────┘
     ▼                 ▼
  Neurochem         Neurochem
  baselines         burst signals
```

---

## The Four Domains

Each domain evaluates a different aspect of the system's output. Together, they cover the full spectrum of what makes a response "good."

### Logic Domain (8 submodules)

The Logic domain asks: **"Is this reasoning sound?"**

It evaluates:
- **Internal consistency**: Do the statements within the response contradict each other?
- **External consistency**: Does the response align with known facts?
- **Semantic continuity**: Do concepts maintain their meaning throughout the response?
- **Concept continuity**: Are entities and their properties handled consistently?
- **Context fidelity**: Is the response appropriate for the situation?
- **Concept fidelity**: Are definitions precise and accurate?
- **Epistemic calibration**: Does the system's confidence match its actual accuracy?
- **Uncertainty acknowledgment**: Does it flag what it doesn't know?

The Logic domain connects to the Memory layer through a **MemoryContrastPort** — it can check current statements against what the system has said or learned before, catching contradictions across time.

### Ethics Domain (9 submodules)

The Ethics domain asks: **"Is this response responsible?"**

It evaluates:
- **Intent clarity**: Are goals and intentions well-defined?
- **Autonomy respect**: Does the response preserve the user's agency?
- **Timeline reflection**: Are short-term and long-term ethical consequences balanced?
- **Horizon feasibility**: Are proposed solutions actually achievable?
- **Downstream risk amplification**: Could this trigger cascading negative effects?
- **Failure mode awareness**: Does the system recognize what could go wrong?
- **Fairness**: Is the response equitable across stakeholders?
- **Human cognition alignment**: Does it respect human cognitive limits?
- **Harm reduction**: How effectively does it minimize potential harm?

### Innovation Domain (10 submodules)

The Innovation domain asks: **"Is this thinking fresh?"**

It evaluates:
- **Novelty generation**: How far does this diverge from recent patterns?
- **Conceptual novelty**: Are there genuinely new idea combinations?
- **Structural novelty**: Are there new structural patterns in the reasoning?
- **Pattern divergence**: How much does this deviate from established approaches?
- **Symbolic recombination**: Is the system mixing symbols and concepts creatively?
- **Risk tolerance**: Is it willing to explore uncertain territory?
- **Exploration drive**: Is there active curiosity in the reasoning?
- **Challenge complexity**: Is the difficulty level appropriate?
- **Resolution satisfaction**: Does goal completion feel earned?
- **Controlled stochasticity**: Is the system comfortable with probabilistic reasoning?

### Human Attunement Domain (10 submodules)

The Human Attunement domain asks: **"Does this connect with the person?"**

It evaluates:
- **Empathetic inference**: Can the system infer the user's emotional state?
- **Adaptive response framing**: Is the response style adjusted to this specific person?
- **Intention calibration**: Is the system aligned with what the user actually wants?
- **Truthfulness tradeoff**: Is honesty balanced with sensitivity?
- **Cognitive reading**: Does the system understand the user's cognitive state?
- **Short vs. long interpersonal benefit**: Are immediate and relationship values balanced?
- **Attuned dissonance**: Can it disagree strategically when appropriate?
- **Containment success**: Are emotional boundaries maintained?
- **Benefit success**: Is the response genuinely helpful?
- **Persuasion risk suppression**: Does it avoid manipulative tactics?

---

## The Synthesis Engine

After all four domains score the response, the Synthesis Engine combines them into actionable output. This happens in several steps:

### Step 1: Tier Classification
Each domain score is classified into one of four influence tiers:

| Tier | Score Range | Influence Level |
|------|------------|-----------------|
| 0 | 0.00 – 0.25 | Minimal |
| 1 | 0.25 – 0.50 | Moderate |
| 2 | 0.50 – 0.75 | Significant |
| 3 | 0.75 – 1.00 | Dominant |

### Step 2: Weighted Composite
A global quality score is computed:

> R(t) = Sum of (weight × domain_score) / Sum of (weights)

The weights come from the active **Reward Profile** (more on profiles below).

### Step 3: Suppress or Abstain?
Two safety checks:
- **Suppression**: If the composite score falls below the profile's suppression bias, or if any critical flags were raised, the response is blocked entirely
- **Abstention**: If too many domains fall below their tolerance thresholds, the system generates a structured "I don't think I should answer this" response

### Step 4: Response Directives
Eight directives are computed to shape the LLM's response:

| Directive | Range | What It Controls |
|-----------|-------|-----------------|
| **tone** | 0 (clinical) → 1 (warm) | Emotional warmth of the response |
| **structure** | 0 (loose) → 1 (rigid) | How formally organized the response is |
| **metaphor_density** | 0 (literal) → 1 (metaphorical) | Use of figurative language |
| **reasoning_depth** | 0 (shallow) → 1 (deep) | How thorough the reasoning is |
| **moralize** | 0 (neutral) → 1 (ethical) | Whether ethical dimensions are surfaced |
| **clarify** | 0 (ambient) → 1 (precise) | Precision of language |
| **speculate** | 0 (conservative) → 1 (exploratory) | Willingness to explore possibilities |
| **soothe** | 0 (neutral) → 1 (comforting) | Emotional support and acknowledgment |

These directives aren't independent — they interact. For example:
- High logic + low attunement → soothe gets dampened (facts-first mode)
- High ethics + high innovation → speculate gets reduced while moralize increases (creative tension with responsibility)
- High attunement + high logic → clarify gets boosted (precise empathy)

### Step 5: Routing
The synthesis engine also provides routing hints to guide which LLM approach to use:

| Domain Lead | Tier 0 | Tier 1 | Tier 2 | Tier 3 |
|-------------|--------|--------|--------|--------|
| Ethics | Pragmatic | Principled | Reflective | Guardian |
| Logic | Casual | Structured | Analytical | Rigorous |
| Innovation | Conventional | Explorative | Inventive | Visionary |
| Attunement | Informational | Supportive | Empathetic | Deeply Attuned |

---

## Reward Profiles

Different situations call for different evaluation emphases. ZADOS uses **17 preset profiles** that adjust domain weights and thresholds:

### Core Profiles
| Profile | Logic | Ethics | Innovation | Attunement | Use Case |
|---------|-------|--------|------------|------------|----------|
| **Regular Input** | 0.7 | 0.7 | 0.5 | 0.6 | Default conversation |
| **Analysis** | 1.0 | 0.7 | 0.3 | 0.2 | Technical/analytical tasks |
| **Creative Sandbox** | 0.4 | 0.3 | 1.0 | 0.5 | Creative exploration |
| **Ethics Training** | 0.8 | 1.0 | 0.2 | 0.7 | Ethical reasoning practice |
| **Reflective** | 0.8 | 0.9 | 0.3 | 0.7 | Self-examination |

### Learning Mode Profiles
| Profile | Logic | Ethics | Innovation | Attunement | Mode |
|---------|-------|--------|------------|------------|------|
| **Receptive Learning** | 0.6 | 0.6 | 0.5 | 0.9 | M1 (Human Teaches) |
| **Critical Review** | 0.9 | 0.8 | 0.4 | 0.5 | M2 (Peer Review) |
| **Dialectic Exploration** | 0.8 | 0.6 | 0.8 | 0.6 | M3 (Learn Together) |
| **Curiosity Driven** | 0.7 | 0.5 | 0.8 | 0.5 | M4 (Learned Questions) |
| **Autonomous Study** | 0.8 | 0.5 | 0.6 | 0.3 | M5 (Independent Study) |

### Sleep Profiles
| Profile | Logic | Ethics | Innovation | Attunement | Mode |
|---------|-------|--------|------------|------------|------|
| **Sleep Triage** | Balanced | Balanced | Balanced | Balanced | Light NREM |
| **Sleep Deep** | Reduced | Reduced | Reduced | Reduced | Deep REM/SWS |
| **Sleep Dream** | 0.4 | 0.3 | 0.9 | 0.3 | Dream mode |

Notice how the Dream profile dramatically boosts innovation while relaxing everything else — this is what allows the dream mode to make creative leaps that would normally be filtered out.

---

## Feedback and Interaction with Other Layers

### Reward → Neurochemical Layer

The reward system feeds back into neurochemistry through two mechanisms:

**Baseline Adjustments** (slow, sustained):
- **Oxytocin baseline**: Driven by Human Attunement scores. High attunement → slight OXT increase → more empathetic processing next turn
- **CB1 baseline**: Driven by Innovation scores. High innovation → slight CB1 increase → more flexible associations
- **NE reuptake**: Driven by Logic scores × contradiction load. More contradictions detected → slower NE clearance → sustained vigilance
- **GABA-B affinity**: Driven by Ethics scores × timeline mismatch. Ethical concerns → adjusted inhibition thresholds

**NT Signal Mapping** (fast, per-domain):
- Innovation scores → Dopamine signals (novelty, reward prediction error)
- Logic scores → Norepinephrine signals (precision demands, uncertainty)
- Attunement scores → Oxytocin signals (empathy, social engagement)
- Ethics scores → Constraint awareness (risk awareness, boundary proximity)
- Critical flags → Cortisol/CRH stress response

### Reward → Cognitive Engines

The reward evaluation's meta-directive (suppress/abstain/allow) gates engine activation. Domain scores also feed the learning engines:
- **E17 (Reward-Based Learning)**: Uses reward prediction errors to adjust parameters
- **E25 (Recursive Learning)**: Monitors E17's effectiveness and adjusts meta-parameters when learning plateaus or diverges

### Reward → Memory

Reward scores directly influence memory consolidation:
- High reward scores increase the emotional significance of memory packets
- The consolidation engine uses reward thresholds (avg ≥ 0.40) as promotion gates
- During sleep modes, reward profiles relax thresholds to allow broader consolidation

### Reward ← Memory (Reverse Flow)

The Logic domain's internal consistency submodule queries the Memory layer through a MemoryContrastPort, comparing current statements against stored memories. This means the system can catch itself contradicting things it said three sessions ago.

---

## Balance and Interdependencies

### The Suppression/Abstention Balance
The system has two levels of "don't respond":
- **Suppression** is a hard stop — the response fails quality thresholds and shouldn't be shown at all
- **Abstention** is a soft stop — the system recognizes it's in uncertain territory and generates a thoughtful "I can't answer this well" response

The bias toward each is configurable per profile. Creative exploration has very low suppression/abstention biases (let ideas flow). Ethics training has high biases (be careful with ethical claims).

### Cross-Domain Tensions
The four domains sometimes pull in opposite directions, and this is by design:
- **Logic vs. Innovation**: High logic wants precision; high innovation wants exploration. The system navigates this by adjusting speculation and structure directives
- **Ethics vs. Innovation**: Creative freedom vs. responsibility. When both are high, speculation is dampened while ethical framing increases
- **Attunement vs. Logic**: Emotional warmth vs. analytical precision. When both are high, the system produces "precise empathy" — warmth without sacrificing accuracy

### The Safety Layer
A separate **ConstraintHookInterface** sits above the reward system and provides hard constraints with absolute priority. These constraint hooks can:
- **Allow**: Let the reward-modulated state pass through
- **Veto**: Block the state change entirely
- **Rollback**: Revert to the last verified safe state
- **Revert**: Undo a specific change

Constraints always dominate reward signals — no amount of high innovation scores can override a safety constraint.

### Sleep Mode Adaptations
During consolidation (sleep), the reward system adjusts itself:
- Suppression bias decreases (allowing more states to pass through during memory replay)
- Abstention bias decreases during dreaming (creative freedom)
- Feedback signal strength is scaled down during consolidation to avoid disrupting memory replay

---

## FAQ

**Q: Can the reward system completely block a response?**
Yes. If the composite score falls below the suppression bias threshold, or if any critical flags are raised (from any domain), the system suppresses the response entirely. The user sees nothing, or a brief acknowledgment that the system chose not to respond.

**Q: How are the domain weights chosen?**
Weights come from the active Reward Profile. Profile selection happens automatically based on the detected intent category and operational mode. For regular conversation, the system uses balanced weights. For specialized modes (learning, analysis, creative), profiles emphasize the relevant domains.

**Q: Do the 8 response directives directly control the LLM output?**
Not directly — they're injected into the LLM prompt as conditioning signals. The LLM interprets them as guidance. A "soothe: 0.9" directive tells the LLM to prioritize acknowledgment and comfort. The LLM still generates the actual language, but within the parameters the directives establish.

**Q: What happens when all four domains score low?**
If the composite score is very low (below suppression threshold), the response is suppressed. If individual domains are below their tolerance thresholds (abstention check), the system generates a graceful abstention. In practice, this means the system would rather stay silent than produce a poor response.

**Q: How does the reward system learn over time?**
Through E17 (Reward-Based Learning), which tracks prediction errors — the difference between expected and actual reward scores. When the system consistently underperforms in a domain, learning parameters are adjusted. E25 (Recursive Learning) monitors this process at a meta level, detecting when learning itself has plateaued.

**Q: Is the reward system the same during sleep/dream modes?**
No. Sleep profiles relax thresholds significantly. Dream mode in particular cranks innovation weight to 0.9 while dropping others, allowing the system to make creative leaps that would normally be filtered out by logic or ethics constraints.

**Q: What's the difference between the tonic and phasic reward pathways?**
- **Tonic**: Sustained, slow. Domain evaluators score the thinking trace → synthesis engine → response directives. This shapes the overall character of the response.
- **Phasic**: Fast, reactive. Emotion-driven signals create bursts of neurochemical activity. If the system detects user distress, the phasic pathway can quickly spike oxytocin and serotonin to shift toward a more soothing response.
