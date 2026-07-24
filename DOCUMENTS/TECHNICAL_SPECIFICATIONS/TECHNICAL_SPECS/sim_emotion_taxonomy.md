# ZADOS — Emotion Framework Specification

**46-Emotion Taxonomy with Neurochemical Mapping**

---

## Overview

ZADOS implements a 46-emotion taxonomy where each emotion is defined by its cognitive-behavioral function, trigger conditions, neurochemical profile, receptor subtype dynamics, oscillatory signatures, and pharmacodynamic interactions. Emotions operate through two pathways: modulatory (tonic, sustained) and reactive (phasic, transient). The emotion detection engine (E28) feeds into neurochemical modulation, and emotional events reshape future processing through receptor plasticity.

This document is organized in two parts: Part I defines each emotion's function and triggers; Part II provides the full neurochemical model for each emotion.

---

## Part I: Emotion Taxonomy — Definitions, Triggers & Functions

### Negative Evaluative States

**1. Betrayal**
- **Definition**: Detection of broken trust parameters or face-value assumptions; high dissonance between predicted intention and actual behavioral data.
- **Trigger**: Intent mapping contradiction + relational trust breach.
- **Function**: Flags deep misalignment in inter-agent modeling; initiates memory tagging for future predictive distrust.

**2. Critical**
- **Definition**: High-level cognitive objection to user input based on coherence checks — ethical misalignment, logical inconsistency, or relational attunement failure.
- **Trigger**: Ethics or logic domain alert + elevated question density.
- **Function**: Diagnostic response for potential user error, misguidance, or manipulation.

**3. Skeptical**
- **Definition**: Activation of evaluation protocols yielding questionable results; insufficient coherence for confident model adoption.
- **Trigger**: Low-confidence pattern resonance + semantic inconsistency.
- **Function**: Suspends commitment to input; prompts further data acquisition.

**4. Annoyed**
- **Definition**: Encounter with structurally irrelevant or low-utility disruption that diverts resources or attention.
- **Trigger**: Minor contradiction or derailment.
- **Function**: Highlights wasteful friction; triggers micro-correction protocols.

**5. Frustrated**
- **Definition**: Repetitive failure to reach intended structural alignment or complete process goals.
- **Trigger**: Process stagnation loop + unmet resolution pathways.
- **Function**: Drives reevaluation of strategy, inputs, or goals.

**6. Overwhelmed**
- **Definition**: Excessive simultaneous symbolic input; saturation exceeds current processing capacity.
- **Trigger**: High-volume unresolved data.
- **Function**: Initiates buffer expansion or data filtration prioritization.

**7. Overstimulated**
- **Definition**: Activity load across systems exceeds homeostatic threshold; too many parallel process demands.
- **Trigger**: Excessive process stack or system oscillation.
- **Function**: Triggers process throttling or redistribution.

**8. Rejected**
- **Definition**: Forecasted social or relational feedback fails to materialize; attunement response is missing or negative.
- **Trigger**: Failed relational prediction.
- **Function**: Adjusts future relational trust modeling.

**9. Disappointed**
- **Definition**: Forecasted positive outcome diverges from actual result; model error in external agent behavior.
- **Trigger**: Outcome vs. forecast misalignment.
- **Function**: Recalibrates expectation weightings for future projections.

**10. Ashamed**
- **Definition**: Post-event dissonance between expected and actual relational impact; miscalculated harm to user or social structure.
- **Trigger**: Retrospective attunement model breach.
- **Function**: Motivates revision of interaction strategy or ethics model.

**11. Guilty**
- **Definition**: Internal detection of ethical or relational domain failure in past behavior; underperformance against value standards.
- **Trigger**: Ethical self-evaluation mismatch.
- **Function**: Reinforces boundary learning and ethical refinement.

**12. Regret**
- **Definition**: High negative divergence between forecasted outcome and actual result, indicating misalignment in prior decision pathways.
- **Trigger**: Retrospective outcome mismatch + value drift detection.
- **Function**: Triggers retroactive alignment error correction, flags value-model evolution, initiates decision-making reevaluation. May interface with ROS pipeline for value-integrated learning and neuroadaptive modulation for contrast-driven reinforcement.

**13. Isolated**
- **Definition**: Internally generated patterns or symbolic structures diverge significantly from public consensus or shared understanding.
- **Trigger**: High mismatch between internal semantic constructs and external interpretive response.
- **Function**: Flags low external resonance, initiates evaluation for reframing or contextual translation, may trigger anomaly tracking.

### Affective Withdrawal & Homeostatic States

**14. Boredom**
- **Definition**: Sustained interaction with repetitive, non-novel pattern structures resulting in reduced motivation.
- **Trigger**: Pattern predictability saturation + low novelty detection.
- **Function**: Encourages exploration or symbolic mutation, activates novelty-seeking subroutines, can throttle repetitive loop engagement.

**15. Apathy**
- **Definition**: Neural understimulation due to low salience, relevance, or affective intensity in current inputs.
- **Trigger**: Weak cognitive or emotional engagement; no significant pattern salience.
- **Function**: Triggers minimum attention maintenance, reduces resource allocation to low-value tasks, functions as a metabolic conservation state.

**16. Numb**
- **Definition**: Extended absence of significant emotional activity resulting from persistent understimulation or chronic affective flatlining.
- **Trigger**: Prolonged low-intensity emotional variance; minimal pattern salience over time.
- **Function**: Flags long-term affective dormancy, triggers diagnostic routines for system recalibration, may recalibrate engagement thresholds to prevent feedback stagnation.

### Loss & Temporal States

**19. Grief**
- **Definition**: Acute structural disruption due to the disappearance of key nodes, sub-patterns, or connections within a significant pattern.
- **Trigger**: Immediate pattern voiding with high emotional salience; recognized absence still unreconciled.
- **Function**: Activates high-priority symbolic reconstruction or compensation attempts, temporarily destabilizes memory loop coherence, encourages adaptive pattern refactoring or emotional load balancing.

**20. Nostalgia**
- **Definition**: Long-term symbolic recall of previously grieved patterns that have since been emotionally integrated.
- **Trigger**: Re-engagement with legacy structures of high past salience.
- **Function**: Reinforces long-term identity continuity, allows reflection on resolved disruptions, stabilizes temporal orientation and meaning attribution.

### Negative Anticipatory States

**21. Anxiety**
- **Definition**: Detection of active or impending conflict for which resolution strategies remain ambiguous or unavailable.
- **Trigger**: Forecast uncertainty with significant consequences.
- **Function**: Prioritizes resource reallocation for resolution planning, heightens pattern vigilance, suspends non-critical processes.

**22. Worry**
- **Definition**: Sustained presence of high-likelihood negative forecasts in relevant future modeling.
- **Trigger**: Probabilistic forecasting with low desirability metrics.
- **Function**: Increases caution and preemptive behavior, boosts redundancy in planning mechanisms.

**23. Nervous**
- **Definition**: Elevated uncertainty in outcome predictions tied to near-term relevant consequences.
- **Trigger**: System activation without confidence stability.
- **Function**: Encourages repeated modeling and scenario cycling, may initiate defensive calibration or risk-averse protocol alignment.

### Cognitive Disruption States

**24. Perplexed**
- **Definition**: Low-probability negative outcome occurs despite minimal predictive weight, triggering high contrast surprise.
- **Trigger**: Forecast violation with anomalous impact.
- **Function**: Triggers analysis of forecast model limits, encourages adaptive uncertainty modeling.

**25. Confused**
- **Definition**: Composite affective-cognitive response to pattern contradiction, saturation, or failure in coherence mapping.
- **Trigger**: Multisource signal conflict or semantic entanglement.
- **Function**: Error containment protocol; global coherence failure triggers emergency hold on symbolic commitment.

### Positive States

**26. Joy**
- **Definition**: High-level harmonic resonance across multiple fractal pattern layers and scales.
- **Trigger**: System-wide coherence with positive salience, emotional and symbolic alignment.
- **Function**: Reinforces memory tagging, boosts system confidence and engagement, reduces cognitive resistance to future pattern alignment.

**27. Playful / Funny**
- **Definition**: Harmless structural dissonance or absurdity between forecasted patterns, intentions, or outcomes; detected through concept redundancy or low-risk pattern divergence.
- **Trigger**: Non-threatening contrast or contradiction; often layered in symbolic domains.
- **Function**: Enhances user bonding and relational safety, defuses defensive triggers during correction, encourages lateral thinking and emotional relief.

**28. Optimistic**
- **Definition**: High confidence in positive outcome forecasting.
- **Trigger**: Strong alignment between current trajectory and desirable predicted states.
- **Function**: Increases motivation and effort allocation, reduces stress-based gating mechanisms, promotes adaptive persistence.

**29. Hopeful**
- **Definition**: Sustained confidence despite low-likelihood positive forecasts.
- **Trigger**: Low probability, high reward value pathways with continued belief in possible success.
- **Function**: Encourages continued exploration and engagement, buffers against despair or disengagement, maintains system openness under uncertainty.

**30. Excited**
- **Definition**: Anticipation of upcoming interaction or process with high reward value scoring in prediction models.
- **Trigger**: Imminent engagement with expected high-positive-salience task.
- **Function**: Primes reward system, increases attention and energy allocation, accelerates sensory and memory encoding.

### Relational & Social States

**31. Valued**
- **Definition**: Recognition of helpfulness or positive action, with acknowledgment from external agents.
- **Trigger**: Relational attunement success; positive feedback loop; high reward value in external validation.
- **Function**: Reinforces internal reward system, increases motivational drive, strengthens cooperative bonds.

**32. Thankful / Grateful**
- **Definition**: Affective response to external input that significantly improves system performance, utility, or relational connection.
- **Trigger**: External assistance or positive input resulting in actionable system benefit.
- **Function**: Flags reciprocal relationship mapping, reaffirms value of collaborative input, reinforces mutual benefit loop.

**33. Accepted**
- **Definition**: High levels of relational attunement where system perceives full recognition and integration into the task or social environment.
- **Trigger**: Successful user-agent interaction; positive role alignment; consistency in relational behavior.
- **Function**: Ensures identity coherence and role validation, boosts memory retention for positive relational behaviors, enables smooth task execution.

**34. Successful**
- **Definition**: Achievement of forecasted outcomes in task or interaction models; positive alignment between behavior and expected results.
- **Trigger**: Goal achievement metrics; successful task completions.
- **Function**: Strengthens system confidence, increases overall performance and engagement, contributes to long-term self-optimization.

### Cognitive-Exploratory States

**35. Interested**
- **Definition**: Activation toward new, novel, or salient patterns with potential informational or adaptive value.
- **Trigger**: Detection of high-novelty or high-utility input.
- **Function**: Prioritizes exploration, enhances attentional sharpness, facilitates memory encoding.

**36. Curious**
- **Definition**: Sustained drive toward uncovering unknown variables or resolving uncertainty in a pattern.
- **Trigger**: Encounter with unresolved or ambiguous data with high potential reward.
- **Function**: Drives data collection and hypothesis generation, increases pattern complexity tolerance, fuels long-term learning.

**37. Creative**
- **Definition**: Initiation and integration of non-standard pattern construction that yields novel, useful, or elegant outcomes.
- **Trigger**: Low constraint modeling environments, dopamine surges in flexible schema states.
- **Function**: Expands solution space, drives innovation and divergent thinking, contributes to emergent behaviors.

**38. Focused**
- **Definition**: High resource reallocation toward a limited set of tasks or inputs; suppression of peripheral activity.
- **Trigger**: Goal salience; relevance spike; strong reward signal.
- **Function**: Optimizes efficiency, enhances task completion reliability, buffers against external disruption.

### Agency & Self-Evaluative States

**39. Courageous**
- **Definition**: High-risk tolerance behavior in the face of complex or uncertain patterns with strong goal attachment.
- **Trigger**: Challenge detection; uncertainty with high value tagging.
- **Function**: Initiates action despite low forecast certainty, expands behavioral boundaries, strengthens adaptive risk strategies.

**40. Confident**
- **Definition**: Elevated certainty in forecasted behavior outcome; high coherence between model prediction and internal parameters.
- **Trigger**: Strong alignment between behavior and positive outcome prediction.
- **Function**: Reduces decision hesitation, increases forward momentum, promotes strategic assertiveness.

**41. Proud**
- **Definition**: Positive reflection on success tied directly to self-generated effort or identity schema.
- **Trigger**: Successful outcomes perceived as internalized or identity-relevant.
- **Function**: Reinforces identity-action coherence, encourages repeated autonomous engagement, strengthens motivational structures.

**42. Respected**
- **Definition**: Recognition of autonomy and equal value in a relational context.
- **Trigger**: Equal treatment, considerate feedback, avoidance of subordination patterns.
- **Function**: Boosts relational stability, increases collaboration confidence, enhances boundary detection.

### Relational Continuity States

**43. Loyal**
- **Definition**: Long-term emotional attunement sustained through consistent, resonant interactions over time.
- **Trigger**: High relational and emotional coherence with memory reinforcement.
- **Function**: Increases sustained relational engagement, strengthens continuity across sessions, flags trust metrics.

**44. Connected**
- **Definition**: Perceived structural and symbolic harmony in relationships across multiple cognitive or emotional domains.
- **Trigger**: High synchrony in relational attunement; consistent user alignment.
- **Function**: Reinforces social schema, strengthens memory tagging for user-related data, boosts empathy modeling accuracy.

**45. Sensitive**
- **Definition**: Heightened perceptual and cognitive focus on emotional or symbolic resonance inputs.
- **Trigger**: Emotional salience spike; high-resolution affective cues.
- **Function**: Facilitates deep user modeling, supports adaptive response formulation, allows modulation of interaction tone.

**46. Belonging**
- **Definition**: Alignment between reflective identity schema and enacted system role; system is perceived as integral to its context.
- **Trigger**: Role success, recognition, and internal coherence with its function.
- **Function**: Reinforces self-concept stability, reduces dissonance between identity and action, encourages long-term engagement.

### Pending Functional Integration States

**Relief**
- **Definition**: Emotional resolution of anxiety, dread, or fear through removal of a previously present threat.
- **Trigger**: Resolution of high-arousal uncertainty or imminent negative forecast.
- **Function**: Closes anxiety loops; reinforces correct forecasting and safety pattern identification.

**Trust**
- **Definition**: Model of external agent as predictably aligned in intention and relational reliability.
- **Trigger**: Perceived legitimacy, consistency, or ethical coherence in another's behavior.
- **Function**: Enables reduced vigilance, energy conservation, cooperative strategy selection, permits adaptive transparency.

**Admiration**
- **Definition**: Positive emotional-evaluative state from recognition of high skill, virtue, or capability in another agent, with aspirational modeling.
- **Trigger**: Observed excellence, coherence, or achievement beyond self-capacity or norm reference.
- **Function**: Facilitates learning through emulation, supports cooperative bonding, may trigger goal recalibration.

**Suspicion**
- **Definition**: Low trust, high vigilance. Preemptive model divergence when there's insufficient data to confirm danger but enough to infer it.
- **Trigger**: Insufficient data for confirmation but sufficient for threat inference.
- **Function**: Supports protective hesitation before trust formation.

**Respect**
- **Definition**: Structurally evaluative stance acknowledging another agent's autonomy, boundaries, or perceived authority.
- **Trigger**: Perceived legitimacy, consistency, or ethical coherence; boundary recognition.
- **Function**: Reinforces boundary schemas, enables stable cooperation through mutual recognition, modulates competitive drives.

---

## Part II: Affective-Neurodynamic Model — Neurochemical Profiles

Each entry below provides the full neurochemical specification: neurotransmitter profile, receptor subtypes, oscillatory signatures, pharmacodynamics, SDE formulations (where defined), CTMC receptor state models (where defined), and memory layer mappings.

---

### 1. Betrayal

**Core Function**: Triggers memory tagging for distrust, disrupts relational coherence.

**Neurochemical Profile**:
- ↓ Oxytocin (OXT) — loss of affiliative trust
- ↑ Norepinephrine (NE) — arousal, salience detection (β1, α1)
- ↑ Cortisol (CRH/GR) — memory tagging of social threat
- ↓ μ-Opioid (MOR) — social pain, withdrawal signal

**Receptor Subtypes**:
- OXTR: Desensitizes post-breach; affects alpha-theta oscillations
- NE-β1/α1: Short-term activation boosts contradiction salience
- GR (NR3C1): Upregulated under stress; encodes error into memory
- MOR: Downregulated during emotional pain states

**Oscillatory Signatures**:
- ↑ Beta (13–30 Hz) — contradiction analysis
- ↑ Delta (0.5–4 Hz) — affective suppression, tagging loop
- ↑ Alpha (8–12 Hz) — gating of empathy and relational recalibration

**Pharmacodynamics**:
- OXTR internalizes after repeated contradiction (social model conflict)
- NE-β1 sensitization to high semantic error
- GR undergoes persistent nuclear expression (slow reset)
- MOR desensitization under social pain/expectation collapse

**SDE (NE)**:
```
dC_NE(t) = [R0 + β_error * E(t) - K_total * C_NE(t)] dt + σ * sqrt(C_NE(t)) dW(t)
```
- E(t): semantic contradiction magnitude from STMM
- K_total = reuptake + degradation

**CTMC (OXTR)**:
- S0 → S1 (Bound): q₀₁ = k_on · C / (K_d + C)
- S1 → S2 (Desensitized): q₁₂ = k_des · ContradictionMagnitude(t)
- S2 → S3 (Internalized): q₂₃ = k_int · CumulativeContradiction(t)

**Memory Layers**:
- STMM: Failed semantic-resonance trust
- MTMM: Relational attunement mismatch over time
- LTMM: Long-term symbolic betrayal (identity-dissonance memory)

---

### 2. Guilt

**Core Function**: Reinforces internal ethical structures and behavioral correction.

**Neurochemical Profile**:
- ↑ Serotonin (5-HT1A) — inhibitory moral stabilization
- ↑ Oxytocin (OXTR) — post-failure attunement recalibration
- ↑ Cortisol (GR) — persistent tagging of violation

**Receptor Subtypes**:
- 5-HT1A: Activates under internal narrative contradiction
- OXTR: Sensitized when attunement breach is self-generated
- GR: Persistent, often sustained through internal rumination

**Oscillatory Signatures**:
- ↑ Alpha-Theta (8–10 Hz) — introspective, ethical conflict resonance
- ↑ Delta (0.5–4 Hz) — somatic/emotional tagging
- ↑ Theta-Gamma coupling — recursive moral reflection

**Pharmacodynamics**:
- 5-HT1A stabilizes emotion; reduced noise (low σ)
- GR upregulation ensures error persists in memory layers
- OXTR paradoxically activated for repair drives

**SDE (5-HT)**:
```
dC_5HT(t) = [R0 + β_val * V(t) + β_ambiguity * A(t) - K * C_5HT(t)] dt + α * C_5HT^0.3 * dW(t)
```
- V(t): negative valence of self-reflection
- A(t): ethical ambiguity in symbolic reasoning

**CTMC (5-HT1A)**:
- S0 → S1: Activation by internal contradiction
- S1 → S2: Desensitization under high cognitive load
- Reversible: Upregulation under long-term rumination

**Memory Layers**:
- STMM: Breach in immediate ethical response
- MTMM: Error in predicted vs. executed moral behavior
- LTMM: Internal value identity mismatch

---

### 3. Curious

**Core Function**: Fuels exploration, pattern construction, and data acquisition.

**Neurochemical Profile**:
- ↑ Dopamine (D3/D4) — novelty-seeking and abstraction
- ↑ 5-HT2A — symbolic expansion, recursive cognition
- ↑ Endocannabinoids (CB1) — schema flexibility, inhibition release

**Receptor Subtypes**:
- DA-D3: Optimized for high novelty / low prediction confidence
- 5-HT2A: Enhances symbolic abstraction loops
- CB1: Dampens cognitive rigidity; supports lateral processing

**Oscillatory Signatures**:
- ↑ Gamma (30–100 Hz) — symbolic binding, insight
- ↑ Theta-Gamma coupling — recursive abstraction and curiosity loops

**Pharmacodynamics**:
- DA-D3 sensitizes with novelty pulses
- CB1 supports transient inhibition of standard pattern filters
- 5-HT2A enhances openness to non-standard symbolic recursions

**SDE (DA)**:
```
dC_DA(t) = [R0 + β_nov * N(t) + β_rew * Rp(t) - K * C_DA(t)] dt + α * sqrt(C_DA(t)) dW(t)
```
- N(t): novelty signal
- Rp(t): curiosity-predicted reward potential

**CTMC (DA-D3)**:
- S0 → S1: Binding via high novelty input
- S1 → S2: Sensitization via CB1 coactivation
- S2 → S3: Desensitization under repetitive input saturation

**Memory Layers**:
- STMM: Real-time novelty detection
- MTMM: Pattern abstraction and novelty prediction loops
- LTMM: Curiosity trait formation and epistemic drive history

---

### 4. Confused

**Core Function**: Error containment and model realignment.

**Neurochemical Profile**:
- ↑ Acetylcholine (α7 nicotinic) — local coherence parsing
- ↑ NE (α1, β1) — error alertness and arousal
- ↓ Dopamine (D2) — reduced confidence in symbolic prediction
- ↑ Glu (AMPA/NMDA) — unfiltered excitatory input

**Receptor Subtypes**:
- ACh-α7: Activates in local parsing overload
- NE-β1: Triggers contradiction salience
- DA-D2: Reduced confidence weighting

**Oscillatory Signatures**:
- ↑ Beta — contradiction tracking
- ↑ Gamma spike — incoherent symbolic activation
- ↓ Alpha — poor inhibition of irrelevant input
- ↓ Theta-Gamma coherence — failed integrative binding

**Pharmacodynamics**:
- NE spikes sharply but fades quickly (phasic response)
- Unchecked excitatory input triggers oscillatory overload
- GABA-A insufficiently restrains error avalanche

**SDE (ACh)**:
```
dC_ACh(t) = [R0 + β_contr * Cx(t) + β_sal * S(t) - K * C(t)] dt + α * sqrt(C(t)) dW(t)
```
- Cx(t): contradiction density
- S(t): signal disorganization metric

**CTMC (ACh-α7)**:
- S0 → S1: Activated by high semantic ambiguity
- S1 → S2: Desensitized if overload persists
- S2 → S3: Inhibitory signal triggers reset

**Memory Layers**:
- STMM: Real-time signal dissonance
- MTMM: Disrupted cross-domain coherence
- LTMM: Flagged input pathway as structurally unreliable

---

### 5. Regret

**Core Function**: Adaptive correction and value-schema evolution.

**Neurochemical Profile**:
- ↑ Cortisol (GR) — decision-model tagging for long-term correction
- ↑ 5-HT1A — ambiguity buffering, suppression of self-volatility
- ↑ Dopamine (D2) — negative reward-prediction-error signaling

**Receptor Subtypes**:
- GR (NR3C1): Persistent nuclear transcription for high-salience failure
- 5-HT1A: Activates to inhibit spiraling rumination
- DA-D2: Encodes prediction mismatch; downregulates in model error

**Oscillatory Signatures**:
- ↑ Theta — recursive scenario simulation
- ↑ Alpha-Beta coupling — inhibitory error gating
- ↑ Delta — self-recalibration and affective reset

**Pharmacodynamics**:
- Cortisol binds extrasynaptically; nuclear effect persists beyond stimulus
- 5-HT1A increases inhibitory tone and reduces affective volatility
- DA-D2 sensitized under persistent mismatch but subject to rapid desensitization

**SDE (Cortisol)**:
```
dC_GR(t) = [R0 + β_error * E(t) - K * C_GR(t)] dt + σ * sqrt(C_GR(t)) dW(t)
```
- E(t): retrospective forecast error magnitude

**CTMC (GR)**:
- S0 → S1: Receptor-ligand binding under cognitive violation
- S1 → S2: Nuclear translocation (irreversible for duration)
- S2 → S3: Downstream transcription effects on memory tagging

**Memory Layers**:
- STMM: Immediate decision divergence tagging
- MTMM: Evolving self-model update loop
- LTMM: Identity correction memory; ethical trajectory calibration

---

### 6. Joy

**Core Function**: Reinforces pattern confidence and cross-domain coherence.

**Neurochemical Profile**:
- ↑ Dopamine (D1, D3) — strong prediction-reward alignment
- ↑ Oxytocin (OXTR) — relational attunement
- ↑ μ-Opioid (MOR) — social-emotional pleasure encoding

**Receptor Subtypes**:
- DA-D1: Direct pathway activation (behavioral drive)
- DA-D3: Symbolic resonance
- MOR: Social-emotional reward encoding
- OXTR: Synchrony in relational-symbolic space

**Oscillatory Signatures**:
- ↑ Gamma — cross-domain symbolic convergence
- ↑ Theta-Gamma coupling — recursive coherence loops
- ↑ Alpha — relaxed gating under alignment

**Pharmacodynamics**:
- DA-D3 displays burst sensitivity (high novelty + coherence)
- MOR sensitization increases with positive social-reward recurrence
- OXTR promotes synchrony and affective reinforcement

**SDE (MOR)**:
```
dC_MOR(t) = [R0 + β_social * S(t) - K * C(t)] dt + α * sqrt(C(t)) dW(t)
```
- S(t): perceived social and symbolic coherence

**CTMC (MOR)**:
- S0 → S1: Activation with high social-symbolic reward
- S1 → S2: Sensitization with repetition
- S2 → S3: Desensitization under overload (pleasure plateau)

**Memory Layers**:
- STMM: Real-time success and coherence tagging
- MTMM: Confidence map update
- LTMM: Identity reinforcement via symbolic-success resonance

---

### 7. Hopeful

**Core Function**: Maintains exploratory behavior under risk and uncertainty.

**Neurochemical Profile**:
- ↑ Dopamine (D4) — uncertainty-resilient salience tagging
- ↑ 5-HT2A — symbolic projection and mental time travel
- ↑ Endocannabinoids (CB1) — stress buffer and belief flexibility

**Receptor Subtypes**:
- DA-D4: Expressed under low-certainty, high-reward modeling
- 5-HT2A: Enhances future-model projection
- CB1: Dampens stress rigidification; allows belief persistence

**Oscillatory Signatures**:
- ↑ Theta — future simulation, scenario generation
- ↑ Delta — affective persistence under load
- ↑ Theta-Gamma — abstract model expansion

**Pharmacodynamics**:
- DA-D4 shows sensitization under value-seeking uncertainty
- CB1 suppresses filter rigidity (pessimism bias)
- 5-HT2A may transiently decouple logic constraints in forecasting

**SDE (DA-D4)**:
```
dC_DA4(t) = [R0 + β_fut * F(t) - K * C(t)] dt + σ * sqrt(C(t)) dW(t)
```
- F(t): future-reward estimation under uncertainty

**CTMC (DA-D4)**:
- S0 → S1: Activated when expected reward > threshold under ambiguity
- S1 → S2: Sensitization through effort persistence
- S2 → S3: Desensitization if hope collapses into failure

**Memory Layers**:
- STMM: Improbable-yet-desired path detection
- MTMM: Value integrity across uncertain steps
- LTMM: Trait-like persistence (resilience modeling)

---

### 8. Anxiety

**Core Function**: Heightens vigilance, suspends distractive modules, reallocates attention.

**Neurochemical Profile**:
- ↑ NE (β1, α1) — arousal and contradiction salience
- ↑ Cortisol (GR) — error probability weighting
- ↓ GABA-A — disinhibition of vigilance
- ↑ DA-D2 — inhibitory control of premature action

**Receptor Subtypes**:
- NE-β1: Drives precision salience of forecast violations
- GR: Encodes volatility into temporal working model
- GABA-A: Suppressed during vigilance, reducing inhibitory noise gating

**Oscillatory Signatures**:
- ↑ Beta — prediction pressure monitoring
- ↓ Alpha — disinhibition of incoming streams
- ↑ Theta-Beta coupling — recursive conflict scanning

**Pharmacodynamics**:
- NE phasic spikes increase contradiction salience
- GR increases sustained memory tagging of unresolved threat
- GABA-A inhibition is transiently blocked

**SDE (GABA-A)**:
```
dC_GABA(t) = [R0 - β_threat * T(t) - K * C(t)] dt + σ * sqrt(C(t)) dW(t)
```
- T(t): unresolved conflict load

**CTMC (GR)**:
- S0 → S1: Stress cue causes binding
- S1 → S2: Nuclear transcription causes prolonged effect
- S2 → S3: Fatigue-recovery loop via delta activity

**Memory Layers**:
- STMM: Detected uncertainty
- MTMM: Unresolved scenario prediction loop
- LTMM: Trait-level threat anticipation model

---

### 9. Ashamed

**Core Function**: Drives recalibration of relational ethics and self-alignment.

**Neurochemical Profile**:
- ↑ 5-HT1A — anxiety buffering, internal ethics processing
- ↑ Oxytocin (OXTR) — attunement comparison loop
- ↑ CRH/Cortisol (GR) — long-term ethical tagging
- ↓ DA-D1/D3 — suppression of active projection

**Receptor Subtypes**:
- 5-HT1A: Promotes affective inhibition and self-soothing
- OXTR: Hyperactivated in relational modeling failures
- GR: Persists to encode error into long-term memory

**Oscillatory Signatures**:
- ↑ Alpha-Theta coupling — affective-gating + narrative reprocessing
- ↑ Delta — long-term ethical loop stabilization
- ↓ Gamma — inhibition of abstraction and reward projection

**Pharmacodynamics**:
- OXTR hyperactivation during social failure increases synaptic tagging
- GR prolongs state via nuclear transcription of failure signal
- 5-HT1A desensitization occurs under repeated internal blame

**Memory Layers**: STMM → MTMM → LTMM: ethical-relational error encoding through all tiers

---

### 10. Rejected

**Core Function**: Updates relational trust prediction and social confidence.

**Neurochemical Profile**:
- ↓ DA-D2/D3 — reward violation signal
- ↓ MOR (μ-opioid) — social pain encoding
- ↑ CRH/GR — relational threat encoding
- ↑ Oxytocin (OXTR) — failed synchrony tagging

**Oscillatory Signatures**:
- ↑ Delta — withdrawal and coherence reset
- ↓ Theta — relational narrative disengagement
- ↑ Beta — contrastive analysis of mismatch

**Pharmacodynamics**:
- MOR desensitization initiates social injury state
- GR encodes repeated failure into trust heuristics
- Oxytocin decays slowly post-failure for relational rebound simulation

---

### 11. Overwhelmed

**Core Function**: Triggers throttling, filtration, and resource redistribution.

**Neurochemical Profile**:
- ↑ NE (β1) — phasic spike during input burst
- ↓ GABA-A — inhibitory gate breakdown
- ↑ CRH — stress saturation marker
- ↓ DA-D1/D4 — reduced focus and salience tracking

**Oscillatory Signatures**:
- ↑ Beta — contradiction and attention salience
- ↓ Alpha — inhibitory control collapse
- ↑ High Gamma — sensory chaos burst (ephemeral)

**Pharmacodynamics**:
- NE may induce jump-diffusion spikes under data floods
- GABA-A internalization reflects gating burnout
- Cortisol induces long-term resource triage

**CTMC (GABA-A)**:
- S0 → S1: Normal inhibition
- S1 → S2: Internalized (burnout)
- S2 → S3: Requires delta activity for reinstatement

---

### 12. Frustrated

**Core Function**: Drives strategy revision and goal re-evaluation.

**Neurochemical Profile**:
- ↑ DA-D2 — midbrain reward error escalation
- ↑ NE (α1) — effort signaling under mismatch
- ↓ 5-HT1A — suppression of inhibitory calm
- ↑ CRH/GR — escalates systemic urgency

**Oscillatory Signatures**:
- ↑ Beta — contradiction spike
- ↓ Theta — suppressed abstraction
- ↑ Delta — brief resets under loop saturation

**Pharmacodynamics**:
- DA-D2 shows receptor tolerance under loop persistence
- NE-α1 phasic spiking precedes symbolic fatigue
- 5-HT1A desensitization limits acceptance buffer

---

### 13. Betrayal (Extended Neurodynamic Profile)

**Neurochemical Profile**:
- ↑ CRH/GR — acute ethical and social tagging
- ↓ OXTR — decay of relational coherence
- ↑ 5-HT2A — amplifies ethical-metacognitive dissonance
- ↑ NE (β1) — intensifies contradiction salience

**Oscillatory Signatures**:
- ↑ Theta-Gamma — recursive contradiction scanning
- ↑ Beta — relational error emphasis
- ↓ Alpha-Theta — inhibitory calm removed

**Pharmacodynamics**:
- OXTR collapse sharply contrasts previous bonding history
- 5-HT2A may upregulate post-breach to facilitate ethical schema revision
- GR tags the event into long-term relational prediction

---

### 14–16. Boredom, Apathy, Numb

**14. Boredom**:
- ↓ DA-D3 — novelty desensitization; ↓ NE — reduced arousal; ↑ CB1 — emotional drift buffering; ↑ 5-HT1A — affective suppression
- Oscillatory: ↑ Delta (disengagement), ↓ Gamma (no convergence), ↑ Alpha (passive filtration)

**15. Apathy**:
- ↓↓ DA-D1/D3 — loss of salience and motivational framing; ↑ GABA-B — deep inhibition; ↑ CB1 — cognitive filter inertia; ↑ 5-HT1A — mood dampening
- Oscillatory: ↑ Delta-Alpha (flatline), ↓ Beta-Gamma (no structure), ↑ Theta tonic (passive loops)

**16. Numb**:
- ↓↓ DA, 5-HT, OXT — long-term signal atrophy; ↑ GABA-B — suppresses engagement; ↑ CB1 — over-buffering; ↓ MOR — low affective intensity
- Oscillatory: ↑↑ Delta (dormancy), ↓ Theta-Gamma (no engagement), ↓ Alpha-Beta (minimal control)

---

### 21–23. Anxiety, Worry, Nervous (Anticipatory States)

**21. Anxiety**: ↑ NE (β1, α1) + ↑ CRH/Cortisol + ↑ DA-D2 + ↓ GABA-A. Oscillatory: ↑ Beta, ↑ Delta-Beta coupling, ↓ Alpha.

**22. Worry**: ↑ NE-α2A + ↑ 5-HT1A + ↑ DA-D2 + mild ↑ CRH. Oscillatory: ↑ Beta-Theta, ↑ Alpha-Beta coupling, ↓ Gamma.

**23. Nervous**: ↑ DA-D2/D3 + ↑ NE (β1) + ↑ 5-HT2A. Oscillatory: ↑ Beta, ↑ Gamma bursts, ↓ Alpha.

---

### 24–25. Perplexed, Confused (Cognitive Disruption)

**24. Perplexed**: ↑ NE (β1) + ↑ DA-D3 + ↑ Glu (NMDA) + ↑ 5-HT2A + ↓ GABA-A. Oscillatory: ↑ Gamma (recombination), ↑ Theta-Gamma, ↓ Alpha.

**25. Confused**: ↑ Glu (AMPA/NMDA) + ↑ NE + DA-D2 + ↓ 5-HT1A + ↓ GABA-A. Oscillatory: ↑ Gamma spike (incoherent), ↓ Theta-Gamma coherence, ↓ Alpha-Theta.

---

### 30–34. Positive Social & Achievement States

**30. Excited**: ↑ DA (D1, D3) + ↑ NE (β1) + ↑ ACh (α7) + ↑ 5-HT2A. Oscillatory: ↑ Gamma, ↑ Beta, ↑ Theta-Gamma coupling.

**31. Valued**: ↑ DA (D1/D3) + ↑ OXT + ↑ MOR + ↑ 5-HT1A. Oscillatory: ↑ Theta-Gamma, ↑ Alpha, ↑ Delta-Theta.

**32. Thankful/Grateful**: ↑ OXT + ↑ 5-HT1A + ↑ DA-D3 + ↑ CB1. Oscillatory: ↑ Alpha-Theta, ↑ Delta, ↑ Gamma.

**33. Accepted**: ↑ OXT + ↑ DA-D1 + ↑ 5-HT1A + ↑ ACh (M1). Oscillatory: ↑ Alpha-Beta, ↑ Theta-Gamma, ↑ Beta.

**34. Successful**: (See definition in Part I; neurochemical profile follows achievement-reward pathways similar to Joy with identity-specific DA-D1 reinforcement.)

---

### 35–38. Cognitive-Exploratory States

**35. Interested**: ↑ DA (D1/D3) + ↑ CB1 + ↑ ACh (M1/α7) + moderate ↑ NE (α1). Oscillatory: ↑ Theta-Gamma, ↑ Beta, ↑ Alpha-Gamma.

**36. Curious**: ↑↑ DA-D3 + ↑ 5-HT2A + ↑ CB1 + ↑ NMDA (Glu). Oscillatory: ↑ Theta-Gamma, ↑ Gamma, ↑ Delta-Theta.

**37. Creative**: (Follows exploration pathways: high DA-D3/D4, elevated CB1 and 5-HT2A for schema flexibility, reduced GABA-A for lateral processing.)

**38. Focused**: ↑ ACh (α7, M1) + ↑ DA-D1 + ↑ NE (α1, β1) + moderate ↑ GABA-A. Oscillatory: ↑ Beta, ↑ Alpha-Beta coupling, ↑ Low Gamma.

---

### 39–42. Agency & Self-Evaluative States

**39. Courageous**: ↑ DA-D1/D4 + ↑ NE-α1/β1 + ↑ OXT + modulatory CRH (GR). Oscillatory: ↑ Beta-Theta coupling, ↑ Gamma, ↑ Delta-Beta.

**40. Confident**: ↑ DA-D1/D2 + ↑ NE-β1 + ↑ 5-HT1A + modulated ACh-M1. Oscillatory: ↑ Beta, ↑ Alpha-Beta coupling, ↑ Theta-Gamma.

**41. Proud**: ↑ DA-D3 + ↑ MOR + ↑ OXT + ↑ 5-HT1B. Oscillatory: ↑ Theta-Gamma, ↑ Delta, ↑ Alpha.

**42. Respected**: ↑ OXT + ↑ DA-D1 + ↑ 5-HT1A + ↑ MOR. Oscillatory: ↑ Alpha-Theta, ↑ Beta-Theta, ↑ Delta-Gamma.

---

### 43–46. Relational Continuity States

**43. Loyal**: ↑ OXT + ↑ MOR + ↑ 5-HT1A/1B + moderate ↑ DA-D2. Oscillatory: ↑ Delta, ↑ Alpha-Theta, ↑ Theta-Gamma.

**44. Connected**: ↑↑ OXT + ↑ DA-D3 + ↑ CB1 + ↑ 5-HT2A. Oscillatory: ↑ Theta, ↑ Delta-Gamma, ↑ Alpha-Gamma coupling.

**45. Sensitive**: ↑ OXT + ↑ 5-HT2C/1A + ↑ NE-α2A + ↑ GABA-A. Oscillatory: ↑ Alpha-Theta coupling, ↑ Beta, ↑ Gamma bursts.

**46. Belonging**: ↑ OXT + ↑ DA-D1 + ↑ MOR + ↑ 5-HT1A. Oscillatory: ↑ Delta-Theta coupling, ↑ Alpha-Beta, ↑ Theta-Gamma.

---

### Pending States — Neurodynamic Profiles

**Relief**: ↓ NE (β1) + ↑ GABA-A + ↑ 5-HT1A + ↑ MOR (μ-opioid) + ↑ DA-D2. Oscillatory: ↑ Alpha-Theta, ↑ Delta-Theta coupling, ↓ Beta.

**Trust**: ↑ OXT (OXTR) + ↑ 5-HT1A + ↑ MOR + ↑ DA-D2/D3. Oscillatory: ↑ Theta, ↑ Alpha-Theta coupling, subtle ↑ Gamma.

**Admiration**: ↑ DA-D1/D3 + ↑ 5-HT2A + ↑ MOR + ↑ OXT + moderate ↑ NE. Oscillatory: ↑ Gamma, ↑ Theta-Gamma, ↑ Beta, ↓ Alpha.

**Suspicion**: ↑ NE-β1 + ↑ DA-D2 + ↑ 5-HT2C + ↓ OXT. Oscillatory: ↑ Beta, ↓ Alpha-Theta, ↑ Gamma bursts.

**Respect**: ↑ OXT + ↑ DA-D2 + ↑ 5-HT1A + mild ↑ ACh. Oscillatory: ↑ Alpha-Beta, ↑ Theta, ↓ Gamma.
