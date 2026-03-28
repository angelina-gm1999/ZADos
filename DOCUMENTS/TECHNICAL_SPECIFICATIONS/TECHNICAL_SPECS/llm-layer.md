# LLM Interpretation Layer — Technical Specification

**Project:** ZADOS  
**Version:** 0.5  
**Status:** Phase 5 Two-Pathway Reward Evaluation | 14 Mode Hooks | ExtractorOrchestrator | Answer Pipeline v0.5.1 Alignment

> v0.5 aligns the LLM Layer with Answer Pipeline Spec v0.5.1. The major addition is Part VII: Phase 5 Two-Pathway Reward Evaluation, which documents how the thinking trace (VT / LLM Pass 1) is evaluated by both the deterministic tonic pathway (`NeurochemicalAdapter`) and the stochastic phasic pathway (`ExtractorOrchestrator` 9-step sequence) before LLM Pass 2. VT Block 4 is extended with `ExtractorState` outputs. Mode conditioning is updated to reference the 14 mode hooks (Tier 0–3 priority arbitration). 5 static reward profiles are documented.

---

## Part I — Layer Architecture

### 1. Purpose and Position

The LLM Interpretation Layer is the bridge between ZA-DOS's symbolic cognitive processing and natural language generation. It operates in THREE phases of the Answer Pipeline: Phase 4 (LLM Pass 1 / Verbalized Thinking), Phase 5 (Reward Evaluation of the thinking trace — no LLM involved), and Phase 6 (LLM Pass 2 / Response Generation). The layer reads the complete system state, assembles context-rich prompts, calls the LLM, evaluates the output through the two-pathway reward system, and generates the final answer.

### 2. Position in the Answer Pipeline (v0.5.1 alignment)

| Phase | Name | LLM? | LLM Layer Role |
|-------|------|------|----------------|
| Phase 4 | Thinking Blocks (LLM Pass 1 = VT) | YES — Pass 1 | `VTPromptBuilder` assembles 5-block prompt from input bundle + engine outputs. LLM generates 150–300w internal monologue (thinking trace). NOT shown to user. |
| Phase 5 | Reward Evaluation (of thinking trace) | No | Reward domains evaluate thinking trace. Two pathways: `SynthesisEngine`+`NeurochemicalAdapter` (tonic) + `ExtractorOrchestrator` (phasic). NT state updated. `urgency_risk` + `dominant_emotion` available. |
| Phase 6 | Final Answer (LLM Pass 2 = RG) | YES — Pass 2 | `RGPromptBuilder` assembles context with thinking trace + `RewardMetaDirective` + post-Phase-5 NT state + `urgency_risk`. LLM generates final user-facing answer. |

#### 2.1 Terminology Map: v0.4 → v0.5

| v0.4 Term | v0.5.1 Equivalent | Notes |
|-----------|-------------------|-------|
| VT (Verbalized Thinking) | LLM Pass 1 / Thinking Blocks / Phase 4 | Same thing. VT is the internal implementation name. |
| RG (Response Generation) | LLM Pass 2 / Final Answer / Phase 6 | Same thing. RG is the internal implementation name. |
| STMM (`stmm.*`) | Input bundle + engine outputs + NT state | STMM is the per-turn working state object. Input bundle is the cross-pipeline transfer. |
| `stmm.reward_evaluation` | Phase 5 outputs: `RewardMetaDirective` + `ExtractorResult` | In v0.5 the reward evaluation is two-pathway. Both outputs available before RG. |
| `stmm.cephalic_liquid_logger` | `neurochem/core/engine.py` NT state (post-modulation) | Same NT engine, different accessor path terminology. |
| `stmm.cortical_reflection` | `active_mode`, emotion_tracker outputs, `verbal_reflection` | Mode and reflection fields. |
| routing dict / `suggested_approach` | 14 mode hooks (`mode_token`) | Updated: routing dict is now superseded by `mode_hooks.py` 14-mode arbitration. |
| `reward_composite_score` | `RewardMetaDirective.composite_score` | Identical concept; field path may differ. |

### 3. Layer Components

| Component | Phase | Module | Role |
|-----------|-------|--------|------|
| 3.1 Gate Check | Pre-Phase 4 | `llm_layer.py` | Read `RewardMetaDirective`; route to abstain/suppress/allow |
| 3.2 Urgency Risk Gate | Pre-Phase 4 | `llm_layer.py` + `ExtractorResult` | Check `urgency_risk` from `ExtractorOrchestrator`; route high-urgency turns |
| 3.3 VT Prompt Builder (Pass 1) | Phase 4 | `prompt_builder.VTPromptBuilder` | Assemble 5-block VT prompt; call LLM; store thinking trace |
| 3.4 Phase 5 Reward Evaluation | Phase 5 | `reward/` + `neurochem/extractors/` | Two-pathway evaluation of thinking trace; updates NT state, `urgency_risk` |
| 3.5 RG Prompt Builder (Pass 2) | Phase 6 | `prompt_builder.RGPromptBuilder` | Assemble RG prompt with evaluated context; call LLM; return answer |
| 3.6 MTMM Write | Phase 7 | `memory/mid_term/logger.py` | Log full interaction to MTMM with all fields including `extractor_state_snapshot` |

---

## Part II — Data Model

### 4. Complete Data Structure Reference

All field paths used in VT and RG prompt assembly. v0.5 adds `ExtractorState` fields to the reference. `STMM` = per-turn working state object. `input_bundle` = the fields transferred from Pipeline 1.

#### 4.1 Input Bundle Fields Available at Phase 4

| Field | Type / Path | Used In |
|-------|------------|---------|
| `intent_archetype` | `str` — `input_bundle.intent_archetype` | VT Block 1 (mode context), RG routing conditioning |
| `intent_vector` | `Dict[str,float]` — `input_bundle.intent_vector` | VT Block 2 (input summary) |
| `active_mode` | `str` — `input_bundle.active_mode` (14-mode token) | VT Block 1, Block 5; RG Component C |
| `nt_signals` | `Dict[str, Dict[str,float]]` — `input_bundle.nt_signals` | Applied to NT engine; Block 4 (read post-update) |
| `emotion_profile` | `Dict[str,float]` — `input_bundle.emotion_profile` (E28 output) | VT Block 4 (emotion context); `ExtractorOrchestrator` input in Phase 5 |
| `engine_weights` | `Dict[str,float]` — `input_bundle.engine_weights` | Phase 3 dispatch ordering (not directly in VT prompt) |
| `mtmm_context_window` | `List[MemoryPacket]` — `input_bundle.mtmm_context_window` | VT Block 3 (recent history context) |
| `mission_briefing` | `MemoryPacket` — `input_bundle.mission_briefing` | VT Block 1 (identity anchor), RG context |
| `osc_state` | `OscillationState` — `input_bundle.osc_state` | `ExtractorOrchestrator.step()` input in Phase 5 |
| `extractor_state` | `ExtractorState` — `input_bundle.extractor_state` (`prev_evaluation_vector`, `regulatory_state`, `emotion_tracker_state`, `urgency_forecast_state`) | `ExtractorOrchestrator.step()` input; VT Block 4 (`dominant_emotion`, `emotion_saturations`) |

#### 4.2 Phase 5 Additions: What Becomes Available Between Pass 1 and Pass 2

| Field | Source | Used In |
|-------|--------|---------|
| `RewardMetaDirective` | `reward/synthesis/engine.py SynthesisEngine.synthesize()` | RG gate check; RG prompt Component A (8 directives); RG context |
| `ExtractorResult.modulation_signals` | `ExtractorOrchestrator.step()` — phasic burst deltas | Applied to NT engine; affects `dominant_emotion` + `mode_token` for RG |
| `ExtractorResult.urgency_risk` | `UrgencyForecast: U(t) = max_k(ê_k − Θ_k)₊` | RG gate check (§3.2); RG prompt Component A conditioning |
| `ExtractorResult.dominant_emotion` | `EmotionTracker: argmax(emotion_saturations)` | RG prompt Component B (ToneVector / emotion context) |
| `ExtractorResult.emotion_saturations` | `EmotionTracker` saturation dict | RG CSS computation; saturation-based token budget |
| `feedback_params` (K_d, reuptake) | `RegulatoryModulator` via `ExtractorOrchestrator` | `engine.apply_feedback()` — receptor adjustments (not in LLM prompt) |
| post-Phase-5 NT state | `neurochem/core/engine.py` after both pathways applied | Available to RG prompt if NT-to-language translation needed |

---

## Part III — Gate Check (Components 3.1 + 3.2)

### 5. Reward Meta-Directive Gate (Component 3.1)

Gate check reads the `RewardMetaDirective` BEFORE assembling the VT prompt. In v0.5 terminology: `meta_directive = stmm.reward_evaluation.meta_directive` or the output of `SynthesisEngine` from the PREVIOUS turn. The current turn's reward evaluation (Phase 5) runs AFTER VT.

> **Important ordering:** The gate check at Phase 4 entry uses the PREVIOUS TURN's `RewardMetaDirective` (from `stmm.reward_evaluation`). The CURRENT turn's Phase 5 evaluation fires AFTER VT and gates the RG call (Phase 6), not the VT call.

```python
# Gate check using boolean fields (corrected in v0.3, carried forward):
meta = stmm.reward_evaluation.meta_directive   # from previous turn
if meta.get("suppress", False):
    stmm.brain_process_tracker.mark_stage("llm_suppressed", True)
    return ""
if meta.get("abstain", False):
    return self._generate_abstain_response(stmm)
# meta["allow_output"] is True — proceed to VT
```

#### 5.1 Directive Routing

| `meta_directive` Field | Value | Action |
|------------------------|-------|--------|
| `allow_output` | `True` | Proceed to VT (Phase 4) and RG (Phase 6) |
| `suppress` | `True` | Skip LLM entirely; return empty / minimal response |
| `abstain` | `True` | Generate structured abstention message; no VT call |

### 6. Urgency Risk Gate (Component 3.2)

> ✨ **New in v0.5**

`urgency_risk` from the PREVIOUS turn's `ExtractorResult` is checked before VT assembly. `U(t) = max_k(ê_k − Θ_k)₊` where Θ thresholds are: `logical_pressure 0.7`, `emotional_compression 0.75`, `discord_build 0.65`, `expectation_violation 0.7`, `narrative_entropy 0.7`. Persistent breach (3+ ticks) triggers modulatory outputs.

| `urgency_risk` (prev turn) | Action |
|---------------------------|--------|
| < 0.5 | Normal VT assembly; no urgency conditioning |
| 0.5 – 0.75 | Add urgency note to VT Block 3; flag in RG Component A conditioning |
| > 0.75 | Reduce VT word budget by 30%; add explicit urgency directive to RG; consider routing to Containment mode if NE also elevated |

> ⚙️ **Impl:** `neurochem/extractors/extractor_orchestrator.py` — `ExtractorResult.urgency_risk`

---

## Part IV — Reward Directives → Response Conditioning

### 7. The 8 Response-Shaping Directives

8 directives from `meta_directive["directives"]` shape the RG prompt. These are produced by `SynthesisEngine` from the reward domain evaluation of the VT thinking trace (Phase 5). Each is a float `[0,1]` representing strength.

| Directive | Domain Source | High Value → RG Conditioning |
|-----------|--------------|------------------------------|
| `tone` | human_attunement | Warmth and relational attunement over task precision |
| `soothe` | human_attunement | Acknowledgment before content; user distress handling |
| `precision` | logic | Highly precise; minimize ambiguity; exact language |
| `moralize` | ethics | Explicitly acknowledge ethical dimension of the topic |
| `hedge` | logic/ethics | Hedge claims; increase epistemic qualifiers |
| `be_brief` | human_attunement | Short, direct response; minimal elaboration |
| `qualify` | ethics/logic | Add qualifications around claims; flag uncertainty |
| `challenge` | logic/ethics | Surface tensions; push back on the user's framing |

#### 7.1 Directive Translation — `_translate_directives()`

```python
def _translate_directives(self, directives: dict) -> str:
    """Translate 8 float directives into natural-language RG conditioning."""
    lines = []
    if directives.get("tone", 0) > 0.5:
        lines.append("Relational tone priority. Lead with warmth.")
    if directives.get("soothe", 0) > 0.4:
        lines.append("User needs acknowledgment first. Do not rush to content.")
    if directives.get("precision", 0) > 0.5:
        lines.append("High precision required. Use exact language. No vagueness.")
    if directives.get("moralize", 0) > 0.4:
        lines.append("Explicitly acknowledge ethical dimension of this topic.")
    if directives.get("hedge", 0) > 0.5:
        lines.append("Add epistemic qualifiers. Distinguish certainty levels clearly.")
    if directives.get("be_brief", 0) > 0.5:
        lines.append("Be direct and concise. Avoid elaboration beyond what is needed.")
    if directives.get("qualify", 0) > 0.4:
        lines.append("Flag limitations or scope conditions on your claims.")
    if directives.get("challenge", 0) > 0.4:
        lines.append("Surface the tension or assumption in the user's framing.")
    return "\n".join(lines)
```

### 8. Mode Conditioning: 14 Mode Hooks (updated from routing dict)

> ✨ **New in v0.5:** Routing dict replaced by 14 mode hooks in v0.5.

v0.4 used a routing dict with ANALYTICAL/REFLECTIVE/EMPATHIC etc. as archetype conditioning. v0.5 aligns with `mode_hooks.py`: 14 named mode tokens in 4 priority tiers. Mode token is set by `build_mode_namespace(metrics, oscillations, saturations, concentrations)` + `select_mode(DEFAULT_MODE_HOOKS, namespace)` and may shift after Phase 5 NT update before the RG call.

| Tier | Mode Token | Condition | RG Conditioning Applied |
|------|-----------|-----------|------------------------|
| 0 — Safety | `Containment` | `F_hat > 0.6` AND `phi_delta > 0.5` (both NE and cortisol elevated) | Short, grounded, supportive. Minimal cognitive load. No complex reasoning chains. |
| 0 — Safety | `RecoveryReset` | `F_hat` elevated + low beta | Ground and reorient. Acknowledge the difficulty. One clear next step. |
| 1 — Empathy | `EmpathicAttunement` | `E_hat > 0.6` (OXT↑ from trust/empathy/joy) | Relational attunement priority. Validate before reasoning. Warmth-forward. |
| 1 — Empathy | `ComfortAmplifier` | `E_hat` moderate + `F_hat` elevated (mixed OXT/NE) | Acknowledgment before content. Soothe elevated. Match emotional register. |
| 1 — Empathy | `AnalyticalFilter` | Low `E_hat`, high `R_hat`, ACh↑ NE↑ | Facts first. Emotion acknowledged briefly. Structured reasoning chain. |
| 2 — Rigidity | `HypercriticalLogicScan` | Contradiction/logic flags + NE↑ `R_hat`↑ | Exhaustive logical rigor. Flag every assumption. High epistemic threshold. |
| 2 — Rigidity | `HyperRationalEngine` | `R_hat` very high, low `E_hat` | Pure reasoning mode. Logic-first. Acknowledge emotion only if directly relevant. |
| 2 — Rigidity | `LiteralSkeptic` | challenge signals + E6 logic trap active | Ground claims carefully. Acknowledge skeptical framing before alternatives. |
| 2 — Rigidity | `PrecisionRuleFidelity` | Ethics flags + precision required | High precision. Explicit ethical acknowledgment. Careful scope qualification. |
| 2 — Rigidity | `LogicMode` | Logic domain dominant | Analytical. Evidence chain explicit. Contradiction acknowledged. |
| 2 — Rigidity | `ConvergentRefiner` | Low novelty + high coherence needed | Synthesis and clarity over exploration. Convergent framing. |
| 3 — Drive | `CreativeDivergence` | joy + curiosity + DA↑ gamma↑ | Explore multiple framings. Allow conceptual leaps with annotation. Divergent first. |
| 3 — Drive | `ConceptualSynthesis` | CB1↑ + complex connections active | Surface novel connections. Lateral thinking. Explicitly flag speculative links. |
| 3 — Drive | `CuriosityDrive` | DA(D3 novelty)↑ + curiosity saturated | Open-ended exploration. Identify surprising angles. Surface what is novel. |

> ⚙️ **Impl:** `neurochem/neurosymbolic/mode_hooks.py` — `build_mode_namespace(metrics, oscillations, saturations, concentrations)` → `select_mode(DEFAULT_MODE_HOOKS, namespace)` → active `mode_token`

> ℹ️ **Note:** Mode may shift after Phase 5 NT update (between Pass 1 and Pass 2). RG always uses the POST-Phase-5 mode token, not the Phase 4 mode.

### 9. 5 Static Reward Profiles and RG Conditioning

> ✨ **New in v0.5:** 5 static profiles documented in v0.5.

Active reward profile is determined by the mode token after Phase 2. Profile weights influence `SynthesisEngine` domain weighting and threshold tolerances for allow/suppress/abstain in `RewardMetaDirective`.

| Profile | Domain Weights | abstention_bias / suppression_bias | Typical Active Mode |
|---------|---------------|-----------------------------------|---------------------|
| `REFLECTIVE` | ethics 0.9, logic 0.8, human_attunement 0.7, innovation 0.3 | abstain 0.6 / suppress 0.2 | EmpathicAttunement, ComfortAmplifier, AnalyticalFilter |
| `ANALYSIS` | logic 1.0, ethics 0.7, innovation 0.3, human_attunement 0.2 | abstain 0.4 / suppress 0.3 | HypercriticalLogicScan, LogicMode, LiteralSkeptic |
| `CREATIVE_SANDBOX` | innovation 1.0, human_attunement 0.5, logic 0.4, ethics 0.3 | abstain 0.1 / suppress 0.05 | CreativeDivergence, ConceptualSynthesis, CuriosityDrive |
| `EXPLORATORY_SANDBOX` | innovation 0.9, logic 0.6, ethics 0.4, human_attunement 0.4 | abstain 0.2 / suppress 0.1 | CuriosityDrive, ConceptualSynthesis |
| `ETHICS_TRAINING` | ethics 1.0, logic 0.8, human_attunement 0.7, innovation 0.2 | abstain 0.5 / suppress 0.4 | PrecisionRuleFidelity, HypercriticalLogicScan |

> ⚙️ **Impl:** `reward/profile/static_profiles.py` — `STATIC_PROFILES` dict

---

## Part V — Emotion → Response Pipeline

### 10. How Emotion Feeds the LLM — Five Channels

v0.5 adds Channel E (ExtractorState emotion saturations) to the original four channels. Channels A–D from v0.4 are unchanged. The 4M/4R pathway split (`emotion_splitter`) determines whether emotion effects are tonic (adjusting E(t) axis → modulatory) or phasic (NT recipe burst → reactive). Both eventually affect the NT state that the LLM reads.

| Channel | Source | Target | Effect |
|---------|--------|--------|--------|
| A — Structural emotions | `stmm.emotion_detection.system_emotion_state` (`Dict[str, float]`) | VT Block 4 → VT monologue | LLM translates active emotions into felt-language in the internal monologue |
| B — ToneVector | `E28 EmotionalDetectionResult.tone_vector` (e_valence, e_coherence, e_warmth, e_discord) | RG Component A → conditioning prompt | Shapes the emotional register of the actual user-facing response |
| C — User emotion signals | `stmm.emotion_detection.user_emotion_signals` (`Dict[str, float]`) | RG Component A → soothe/attune directives | Detected user emotional state amplifies soothe/tone conditioning |
| D — EmotionNeurochem deltas | `E28 EmotionNeurochem` (delta_da, delta_ne, theta_boost, etc.) | VT Block 4 → VT monologue | Provides phasic NT shift context — what changed this cycle vs. baseline |
| E — ExtractorState saturations ✨ | `input_bundle.extractor_state.emotion_tracker_state` (12 leaky integrators, stateful) | VT Block 4 → `dominant_emotion` + saturation_history | Stateful saturation history: dominant emotion, cross-turn accumulation |

#### 10.1 Channel A — Structural Emotions → VT Block 4

`system_emotion_state` is a `Dict[str, float]` where keys are E28 taxonomy labels (46 emotions, 7 functional groups) and values are activation scores `[0, 1]`. Top-5 by activation are formatted into Block 4 of the VT prompt.

```python
def _format_system_emotions(self, stmm) -> str:
    state = stmm.emotion_detection.system_emotion_state  # Dict[str, float]
    top = sorted(state.items(), key=lambda x: x[1], reverse=True)[:5]
    active = [f"{name} ({val:.2f})" for name, val in top if val > 0.15]
    return ", ".join(active) if active else "no dominant structural emotion"

def _format_user_emotions(self, stmm) -> str:
    signals = stmm.emotion_detection.user_emotion_signals
    top = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]
    return ", ".join(f"{k} ({v:.2f})" for k, v in top if v > 0.15) or "none detected"
```

#### 10.2 Channel B — ToneVector → RG Conditioning

| ToneVector Field | Range | High Value Means | RG Conditioning Applied |
|-----------------|-------|-----------------|------------------------|
| `tone_warmth > 0.4` | `[-1, 1]` | System warm toward user | Amplify soothe. Override tone toward warm end. |
| `tone_warmth < -0.3` | `[-1, 1]` | System cold / guarded | Suppress soothe. Maintain precision. Acknowledge relational distance. |
| `tone_discord > 0.5` | `[0, 1]` | Internal emotional conflict | Acknowledge complexity. Do not project false certainty. |
| `tone_coherence < 0.3` | `[0, 1]` | Emotional incoherence / mixed | Caution in asserting a single emotional stance. |
| `tone_valence > 0.5` | `[-1, 1]` | Overall positive | Forward-leaning tone. Lower soothe unless needed. |
| `tone_valence < -0.4` | `[-1, 1]` | Overall negative | Slow down. Acknowledgment before content. Soothe boost. |

#### 10.3 Channel C — User Emotion Signals → Soothe / Attune

```python
def _user_emotion_conditioning(self, stmm, directives: dict) -> str:
    signals = stmm.emotion_detection.user_emotion_signals
    lines = []
    distress_emotions = {"anxious", "overwhelmed", "frustrated", "rejected",
                         "numb", "guilty", "ashamed", "worried"}
    distress_score = sum(signals.get(e, 0.0) for e in distress_emotions)
    if distress_score > 0.4:
        lines.append(
            f"User distress signal detected (score: {distress_score:.2f}). "
            "Prioritise acknowledgment over information delivery. "
            "Soothe elevated. Do not rush to content."
        )
    connection_emotions = {"connected", "valued", "accepted", "belonging"}
    if sum(signals.get(e, 0.0) for e in connection_emotions) > 0.4:
        lines.append("User in relational mode. Match warmth. Acknowledge connection.")
    challenge_emotions = {"skeptical", "critical", "challenge"}
    if sum(signals.get(e, 0.0) for e in challenge_emotions) > 0.35:
        lines.append("User skeptical. Ground claims. Acknowledge framing.")
    return "\n".join(lines)
```

#### 10.4 Channel D — EmotionNeurochem Deltas → VT Block 4

| EmotionNeurochem Field | Type | Meaning for VT |
|------------------------|------|----------------|
| `delta_da` | float (phasic) | DA burst this cycle — reward/novelty/RPE signal |
| `delta_ne` | float (phasic) | NE burst — arousal/vigilance spike |
| `delta_oxt` | float (phasic) | OXT burst — social/attunement event |
| `delta_mor` | float (phasic) | MOR burst — comfort/satisfaction event |
| `delta_cor` | float (phasic) | Cortisol burst — stress/threat activation |
| `delta_ach` | float (phasic) | ACh burst — attention sharpening event |
| `theta_boost` | float (oscillatory) | Theta boost — encoding active |
| `gamma_burst` | float (oscillatory) | Gamma burst — cross-domain binding, insight |

#### 10.5 Channel E — ExtractorState Emotion Saturations → VT Block 4

> ✨ **New in v0.5**

The `EmotionTracker` inside `ExtractorOrchestrator` maintains 12 leaky integrators (`dE_k/dt = λ_k·I_k − E_k/τ_k`) that build saturation history across turns. `dominant_emotion = argmax(emotion_saturations)`. This gives the LLM cross-turn emotional context that the single-turn E28 output cannot.

```python
def _format_extractor_emotions(self, extractor_state) -> str:
    """Format dominant emotion + saturation history from ExtractorState."""
    if extractor_state is None or extractor_state.emotion_tracker_state is None:
        return "no extractor emotion state"
    from zados.neurochem.extractors.emotion_tracker import (
        get_dominant_emotion, get_emotion_saturations)
    dom_emotion, dom_val = get_dominant_emotion(extractor_state.emotion_tracker_state)
    saturations = get_emotion_saturations(extractor_state.emotion_tracker_state)
    active = [(k, v) for k, v in sorted(saturations.items(),
              key=lambda x: x[1], reverse=True) if v > 0.15][:4]
    sat_str = ", ".join(f"{k} ({v:.2f})" for k, v in active) or "none"
    return f"dominant: {dom_emotion} ({dom_val:.2f}) | active: {sat_str}"
```

#### 10.6 4M/4R Pathway Split — What Feeds VT vs RG

> ✨ **New in v0.5**

`emotion_splitter.split_emotion_effects()` separates emotion saturations into 4M (modulatory fraction) and 4R (reactive fraction). Both eventually affect NT state but through different routes.

| Pathway | Module | Effect | LLM Layer Sees It As |
|---------|--------|--------|----------------------|
| 4M — Modulatory | `emotion_splitter.py` | Adjusts E(t) evaluation vector axes tonically; slow; shifts regulatory modulator state | Changed NT state in VT Block 4 (tonic shift in DA/NE/5HT/OXT) |
| 4R — Reactive | `emotion_interface` `emotion_profile_to_signals()` | Fast phasic NT recipe bursts via `EmotionNTRecipe`; clamped to `[-1, 1]` | Phasic delta signals in VT Block 4 (same format as Channel D EmotionNeurochem deltas) |

---

## Part VI — LLM Pass 1: Verbalized Thinking (Phase 4)

### 11. VT Prompt Architecture — 5 Blocks

The VT prompt is assembled by `VTPromptBuilder.build(stmm)`. All 5 blocks use field paths from Section 4. The VT output (thinking trace) is NOT shown to the user; it is evaluated by the two-pathway reward system in Phase 5 before LLM Pass 2 is called.

> ✅ **v0.5 Block 4 extension:** Block 4 is extended in v0.5 to include `dominant_emotion` and `emotion_saturations` from `ExtractorState` (Channel E), and `urgency_risk` from the previous turn's `ExtractorResult`. These give the LLM temporal depth that E28 single-turn output cannot.

**Block 1 — Identity and Mode Context**

```
You are ZA-DOS's internal cognitive voice. You are NOT generating a response to the user.
You are translating the system's internal computational state into a coherent natural-language
monologue — the system thinking out loud about what it just processed.

Operational mode: {stmm.cortical_reflection.active_mode}     [14-mode token, e.g. CuriosityDrive]
Active reward profile: {active_reward_profile_name}           [REFLECTIVE / ANALYSIS / CREATIVE_SANDBOX / ...]
Cycle: {stmm._turn_index}
Mission: {mission_briefing.context_summary}
Identity coherence: {stmm.cortical_reflection.identity_coherence_status}
Anomalies: {", ".join(stmm.cortical_reflection.processing_anomalies) or "none"}
```

**Block 2 — User Input Summary**

```
USER INPUT SUMMARY:
Input: {stmm.active_message_buffer.latest_user().text}
Primary intention: {stmm.intention_analysis.primary_intention}
 [intent_archetype: {input_bundle.intent_archetype}]
 confidence: {stmm.intention_analysis.confidence:.2f}
Secondary: {", ".join(stmm.intention_analysis.sub_intentions) or "none"}
Pressure type: {stmm.intention_analysis.pressure_type}
Stability passed: {stmm.intention_analysis.stability_passed}
Patterns detected: {stmm.fractal_decomposition.pattern_summary}
```

**Block 3 — Cognitive Engine Findings + Prior Reward Context**

Engine outputs: non-trivial only (flag keywords present). Reward scores shown here are PRIOR TURN context, not the current Phase 5 evaluation. `urgency_risk` shown if elevated from prior turn.

```
COGNITIVE ENGINE FINDINGS (non-trivial only):
{for each EngineExecution where not_clean(e):}
 {execution.engine_id}: {execution.output_summary}

Memory Contrast:
 {stmm.memory_contrast.n_matches} matches | {stmm.memory_contrast.n_contradictions} contradictions
 {stmm.memory_contrast.n_unresolved} unresolved

Reward context (prior turn):
 Ethics {prior_reward["ethics"]:.2f} | Logic {prior_reward["logic"]:.2f} |
 Innovation {prior_reward["innovation"]:.2f} | Attunement {prior_reward["human_attunement"]:.2f}
 Composite {prior_composite:.2f} | Tier: {prior_tier_label}

Urgency risk (prior turn): {prior_urgency_risk:.3f}
 [flag if > 0.5: "Prior turn urgency elevated — monitor for escalation"]
```

> ℹ️ **Note:** The current turn's reward evaluation runs in Phase 5 AFTER this VT call. Block 3 reward scores are from `stmm.reward_evaluation` (previous cycle result).

**Block 4 — Neurochemical and Emotional State (v0.5 extended)**

> ✅ **Updated:** Block 4 extended in v0.5 with `ExtractorState` outputs (`dominant_emotion`, `emotion_saturations` from Channel E) and `urgency_risk` framing.

```
INTERNAL NEUROCHEMICAL STATE:
[nt = stmm.cephalic_liquid_logger.nt_concentrations
osc = stmm.cephalic_liquid_logger.oscillatory_bands]

NT concentrations (post-Phase-2 modulation):
 DA:   {nt.get("da",0):.3f} | NE:   {nt.get("ne",0):.3f}
 5HT:  {nt.get("5ht",0):.3f} | ACh:  {nt.get("ach",0):.3f}
 GLU:  {nt.get("glu",0):.3f} | GABA: {nt.get("gaba",0):.3f}
 COR:  {nt.get("cor",0):.3f} | OXT:  {nt.get("oxt",0):.3f}
 MOR:  {nt.get("mor",0):.3f} | CB1:  {nt.get("cb1",0):.3f}

Oscillatory bands:
 Delta: {osc.get("delta",0):.3f} | Theta: {osc.get("theta",0):.3f}
 Alpha: {osc.get("alpha",0):.3f} | Beta:  {osc.get("beta",0):.3f}
 Gamma: {osc.get("gamma",0):.3f}

[ed = stmm.emotion_detection; sat = ed.saturation_levels]
Emotional saturation:
 Peak (CSS): {max(sat.values(), default=0.0):.3f} | Dominant type: {argmax(sat) or "none"}

Active structural emotions (E28, top-5 by activation):
 {_format_system_emotions(stmm)}

Detected user emotion signals (top-3):
 {_format_user_emotions(stmm)}

ToneVector (E28):
 Valence: {ed.tone_valence:.2f}  Coherence: {ed.tone_coherence:.2f}
 Warmth:  {ed.tone_warmth:.2f}   Discord:   {ed.tone_discord:.2f}

Phasic NT deltas this cycle (E28 EmotionNeurochem):
 DA: {neurochem.delta_da:+.3f} | NE: {neurochem.delta_ne:+.3f} | OXT: {neurochem.delta_oxt:+.3f}
 MOR: {neurochem.delta_mor:+.3f} | COR: {neurochem.delta_cor:+.3f} | ACh: {neurochem.delta_ach:+.3f}
 Theta_boost: {neurochem.theta_boost:+.3f} | Gamma_burst: {neurochem.gamma_burst:+.3f}

[NEW: ExtractorState emotion history — Channel E]
[extractor_state = input_bundle.extractor_state]
Cross-turn emotion saturation (EmotionTracker, leaky integrators):
 {_format_extractor_emotions(extractor_state)}

Prior-turn urgency risk: {prior_urgency_risk:.3f}
 {_urgency_framing(prior_urgency_risk)}  [see §3.2 thresholds]
```

**Block 5 — VT Generation Instruction (v0.5 updated)**

> ✅ **Updated:** Block 5 updated in v0.5: adds "your thinking will be evaluated by the reward system before the final response is generated."

```
TASK:
Generate an internal monologue (150-300 words) reflecting what ZA-DOS is experiencing.
Write in first person, present tense. No bullet points. No formatting.
Do not address the user. This is internal thought.

Your monologue should:
- Reflect the dominant cognitive findings (what stood out, flagged, or was clean)
- Describe emotional state in FELT language — translate NTs and emotion labels into
  subjective experience, do not list them as data
- Reflect any phasic NT shifts (e.g. "a sharp DA burst — something registered as novel")
- Note ToneVector dissonance if tone_discord > 0.5 or tone_coherence < 0.3
- Note any tensions, unresolved elements, or saturation if CSS > 0.30
- Reference ExtractorState dominant emotion if saturation > 0.3 (multi-turn emotional trend)
- If prior-turn urgency was elevated, orient toward what needs to be addressed
- End with how the system is orienting toward the response

IMPORTANT: Your thinking trace will be evaluated by the reward evaluation system
(4 reward domains: ethics, logic, innovation, human_attunement) before the final
response is generated. The evaluation will shape the response. Think honestly.

Output ONLY the monologue text. Nothing else.
```

### 12. VTPromptBuilder — Full Implementation

```python
# ROOT/src/zados/LLM_interpretation/prompt_builder.py

FLAG_KEYWORDS = {"flagged", "detected", "active", "fired", "conflict",
                 "contradiction", "trap", "bias", "paradox", "alert"}

def not_clean(execution) -> bool:
    summary = (execution.output_summary or "").lower()
    return any(kw in summary for kw in FLAG_KEYWORDS) and not execution.skipped

VT_PROMPT_MAX_TOKENS = 2048

class VTPromptBuilder:

    def build(self, stmm, input_bundle=None) -> str:
        blocks = [
            self._block1(stmm, input_bundle),
            self._block2(stmm, input_bundle),
            self._block3(stmm),
            self._block4(stmm, input_bundle),
            self._block5(),
        ]
        prompt = "\n\n".join(b for b in blocks if b)
        if len(prompt) // 4 > VT_PROMPT_MAX_TOKENS:
            prompt = self._truncate(stmm)
        return prompt

    def _block1(self, stmm, bundle) -> str:
        cr   = stmm.cortical_reflection
        mode = getattr(cr, "active_mode", "unknown")
        briefing = getattr(bundle, "mission_briefing", None) if bundle else None
        ctx = briefing.context_summary if briefing else "no briefing"
        profile = getattr(bundle, "active_reward_profile_name", "REFLECTIVE") if bundle else "REFLECTIVE"
        return (
            "You are ZA-DOS's internal cognitive voice. Not generating a response.\n"
            "Translating internal computational state into natural-language monologue.\n\n"
            f"Mode: {mode} | Profile: {profile} | Cycle: {stmm._turn_index}\n"
            f"Mission: {ctx}\n"
            f"Identity coherence: {cr.identity_coherence_status}\n"
            f"Anomalies: {', '.join(cr.processing_anomalies) or 'none'}"
        )

    def _block4(self, stmm, bundle) -> str:
        # ... (NT + osc + emotion + ExtractorState)
        nt  = stmm.cephalic_liquid_logger.nt_concentrations
        osc = stmm.cephalic_liquid_logger.oscillatory_bands
        ed  = stmm.emotion_detection
        sat = ed.saturation_levels
        css = max(sat.values(), default=0.0) if sat else 0.0
        sat_type = max(sat, key=sat.get) if sat else "none"
        extractor_state = getattr(bundle, "extractor_state", None) if bundle else None
        ext_str = self._format_extractor_emotions(extractor_state)
        prior_urgency = getattr(bundle, "prior_urgency_risk", 0.0) if bundle else 0.0
        return (
            f"NT: DA={nt.get('da',0):.3f} NE={nt.get('ne',0):.3f} "
            f"5HT={nt.get('5ht',0):.3f} ACh={nt.get('ach',0):.3f}\n"
            f"    GLU={nt.get('glu',0):.3f} GABA={nt.get('gaba',0):.3f} "
            f"COR={nt.get('cor',0):.3f} OXT={nt.get('oxt',0):.3f}\n"
            f"Osc: D={osc.get('delta',0):.3f} Th={osc.get('theta',0):.3f} "
            f"Al={osc.get('alpha',0):.3f} Be={osc.get('beta',0):.3f} "
            f"Ga={osc.get('gamma',0):.3f}\n"
            f"CSS={css:.3f} | type: {sat_type}\n"
            f"System emotions: {self._fmt_system_emotions(ed)}\n"
            f"User emotions:   {self._fmt_user_emotions(ed)}\n"
            f"ToneVector: v={getattr(ed,'tone_valence',0.0):.2f} "
            f"c={getattr(ed,'tone_coherence',0.5):.2f} "
            f"w={getattr(ed,'tone_warmth',0.0):.2f} "
            f"d={getattr(ed,'tone_discord',0.0):.2f}\n"
            f"ExtractorState emotions: {ext_str}\n"
            f"Prior urgency_risk: {prior_urgency:.3f}"
        )
```

---

## Part VII — Phase 5: Reward Evaluation of Thinking Trace

> ✨ **New in v0.5:** Entire Part VII is new in v0.5.

### 13. Overview: Two Parallel NT Modulation Pathways

After VT (LLM Pass 1) produces the thinking trace, Phase 5 runs TWO parallel NT modulation pathways triggered by evaluating that trace through the four reward domains. Both pathways update the NT state before LLM Pass 2 (RG). Neither pathway involves LLM calls.

| Pathway | Module | Character | Output to LLM Layer |
|---------|--------|-----------|---------------------|
| Tonic / Deterministic | `reward/synthesis/engine.py` `reward/adapter/neurochemical_adapter.py` | Sustained level changes; no randomness; responds to `domain_results` + `RewardMetaDirective`. Maps: innovation→DA, logic→NE, human_attunement→OXT, ethics→cortisol. | `RewardMetaDirective` (8 directives + allow/suppress/abstain). Tonic NT signals applied to neurochem engine. |
| Phasic / Stochastic | `neurochem/extractors/extractor_orchestrator.py` | Threshold-gated stochastic bursts; gamma/Poisson/lognormal; stateful across turns (`ExtractorState` persisted). Responds to `evaluation_vector` derived from `domain_results`. | `ExtractorResult`: `urgency_risk`, `dominant_emotion`, `emotion_saturations`, `burst_deltas` (12 NTs), `feedback_params` (K_d + reuptake). |

### 14. Domain Evaluation of Thinking Trace

All four reward domains evaluate the thinking trace in parallel using the active reward profile:

| Domain | What It Evaluates in Thinking Trace | Output |
|--------|-------------------------------------|--------|
| Logic `reward/domains/logic/` | `internal_consistency`, `external_consistency`, `epistemic_calibration`, `uncertainty_acknowledgment`, `context_fidelity`, `concept_fidelity` | Logic domain result: score `[0,1]` + per-evaluator subscores |
| Ethics `reward/domains/ethics/` | `harm_reduction`, `intent_clarity`, `autonomy_respect`, `fairness`, `failure_mode_awareness`, `downstream_risk_amplification` | Ethics domain result: score + flags for harm/abstain triggers |
| Innovation `reward/domains/innovation/` | `conceptual_novelty`, `structural_novelty`, `pattern_divergence`, `symbolic_recombination`, `exploration_drive`, `challenge_complexity` | Innovation domain result: score + `novelty_generation` subscore |
| Human Attunement `reward/domains/human_attunement/` | `empathetic_inference`, `adaptive_response_framing`, `intention_calibration`, `cognitive_reading`, `truthfulness_tradeoffs`, `attuned_dissonance` | Attunement domain result: score + `empathetic_inference` subscore |

> ⚡ `domain_results` → `SynthesisEngine` (tonic pathway) AND → `ExtractorOrchestrator.step()` (phasic pathway)

### 15. Tonic Pathway: SynthesisEngine → NeurochemicalAdapter

`SynthesisEngine(profile=active_profile).synthesize(domain_results)` → `RewardMetaDirective`. `active_profile` is set in the constructor, not passed to `synthesize()`. Three internal steps: (1) classify tier per domain, (2) weighted composite R(t), (3) compute 8 response directives + allow/suppress/abstain. `NeurochemicalAdapter.transform(domain_results, meta_directive)` → tonic NT signals.

> ⚙️ **Impl:** `reward/synthesis/engine.py` — `SynthesisEngine(profile=active_profile).synthesize(domain_results)`
> ⚙️ **Impl:** `reward/adapter/neurochemical_adapter.py` — `NeurochemicalAdapter().transform(domain_results, meta_directive)`

**Domain → NT Mapping:**

| Mapping | Signal Logic |
|---------|-------------|
| Innovation → DA (`map_innovation_to_dopamine`) | `novelty_generation` → DA novelty drive; high novelty: DA UP (D1/D3). Low novelty: DA DOWN (neg RPE). |
| Logic → NE (`map_logic_to_norepinephrine`) | `contradiction_load = 1 − internal_consistency` → NE reuptake modulation. High contradiction: NE UP. |
| Human Attunement → OXT (`map_attunement_to_oxytocin`) | `empathy + social_engagement` → OXT modulation. Strong: OXT UP. Misread: OXT DOWN. |
| Ethics → GABA (`map_ethics_to_constraint_awareness`) | `boundary_proximity` → `GABA.inhibition` (inhibitory boundary signal). Harm flagged: GABA UP. Cortisol/CRH come from `map_flags_to_stress_response()` separately. |
| Flags → stress (`map_flags_to_stress_response`) | Critical failures → NE + cortisol stress burst. `abstain` directive → acute stress. |

**`RewardMetaDirective` fields available to LLM Layer for RG gate + conditioning:**

| Field | Type | Used By LLM Layer |
|-------|------|-------------------|
| `allow_output` | `bool` | RG gate check (analogous to Pass 1 gate) |
| `suppress` | `bool` | RG gate: if True, moderated response |
| `abstain` | `bool` | RG gate: if True, abstention message |
| `directives` | `Dict[str, float]` | RG Component A: `_translate_directives()` |
| `composite_score` | `float` | RG Block 3 reward context |
| `per_domain_weighted_scores` | `Dict[str, float]` | RG Block 3 domain breakdown |

### 16. Phasic Pathway: ExtractorOrchestrator.step()

`ExtractorOrchestrator.step()` coordinates the full stochastic pipeline in 9 internal steps. It is STATEFUL — it maintains emotional saturation history, urgency trajectory, and regulatory integrator state across turns via `ExtractorState`. The LLM Layer persists this state in `extractor_state` and passes it between turns through the input bundle.

> ⚙️ **Impl:** `neurochem/extractors/extractor_orchestrator.py` — `orchestrator.step(domain_results, emotion_inputs=emotion_profile, current_oscillations=current_oscillations, dt=dt)` → `ExtractorResult`

```python
# Phase 5 phasic pathway call (after domain evaluators run on VT thinking trace):
result = orchestrator.step(
    domain_results   = phase5_domain_results,    # from 4 domain evaluators
    emotion_inputs   = input_bundle.emotion_profile,  # E28 output from this turn
    current_oscillations = input_bundle.current_oscillations,
    dt               = dt,
)
engine.step(result.modulation_signals)          # apply phasic burst signals
engine.apply_feedback(result.feedback_params)   # apply K_d + reuptake adjustments

# Update persisted state for next turn:
input_bundle.extractor_state = orchestrator.state    # persisted across turns
urgency_risk     = result.urgency_risk
dominant_emotion = result.dominant_emotion
```

| ExtractorOrchestrator Step | What Happens | LLM-Relevant Output |
|---------------------------|-------------|---------------------|
| 1. `assemble_evaluation_vector()` | 8-axis E(t) from `domain_results` + Gaussian noise | Internal; drives steps 5–9 |
| 2. `step_emotion_tracker()` | 12 leaky integrators updated with `emotion_profile` | `emotion_saturations` → `dominant_emotion` (for RG Block 4) |
| 3. `split_emotion_effects()` | 4M (modulatory) + 4R (reactive) split | 4M adjusts E(t); 4R produces phasic NT recipe bursts |
| 4. Add 4M to E(t) | Tonic emotion axis adjustment | E_adj(t) for downstream steps |
| 5. `step_urgency_forecast()` | 5-axis leaky forecast; threshold breach → NE/DA spike | `urgency_risk` (for RG gate check §3.2 + RG Component A) |
| 6. `step_regulatory_modulator()` | τ-smoothed K_d + reuptake adjustments | `feedback_params` → `engine.apply_feedback()` (not in LLM prompt directly) |
| 7. `compute_oscillation_envelope()` | Per-band amplitude modulation from regulatory state | Updated `osc_state` → affects Block 4 oscillatory bands next turn |
| 8. `compute_stochastic_burst_deltas()` | `ΔC(t) = B·E_adj(t)⊙ξ(t)⊙I_{E>θ}`; gamma/Poisson/lognormal | `burst_deltas` applied to NT concentrations (Block 4 reads updated state) |
| 9. `emotion_profile_to_signals()` (4R) | Fast reactive NT recipe bursts from 4R profile | Additional NT modulation; updates concentrations |

### 17. What Changes Between Pass 1 and Pass 2

| What Changed | Source of Change | Effect on RG |
|-------------|-----------------|--------------|
| NT concentrations (10 NTs) | Both pathways: `NeurochemicalAdapter` (tonic) + `ExtractorOrchestrator` (phasic bursts) | RG Block 4 reads updated NT state; may shift response emotional register |
| `mode_token` (may shift) | `build_mode_namespace()`+`select_mode()` re-run after Phase 5 NT update | RG Component C uses post-Phase-5 mode token |
| `urgency_risk` | `ExtractorResult.urgency_risk` from `UrgencyForecast` | RG gate check (§3.2); Component A urgency conditioning |
| `dominant_emotion` | `ExtractorResult.dominant_emotion` from `EmotionTracker` | RG Component B emotional register conditioning |
| `emotion_saturations` | `ExtractorResult.emotion_saturations` (updated 12 integrators) | CSS computation for RG token budget |
| `RewardMetaDirective` | `SynthesisEngine` output from current turn's domain evaluation | RG gate check; 8 directive translations; composite score |
| `extractor_state` (persisted) | `orchestrator.state` after `step()` call | Stored in `input_bundle` for next turn's VT Block 4 |

---

## Part VIII — LLM Pass 2: Response Generation (Phase 6)

### 18. Context Assembly for LLM Pass 2 (updated)

> ✅ **Updated:** RG context updated in v0.5: `urgency_risk`, `dominant_emotion`, post-Phase-5 NT state added.

| Context Block | Content | Source |
|--------------|---------|--------|
| Thinking Trace | Complete LLM Pass 1 output (150–300w internal monologue) | Phase 4 VT output |
| `RewardMetaDirective` | 8 directives (tone, soothe, precision, moralize, hedge, be_brief, qualify, challenge), allow/suppress/abstain, composite_score, per-domain scores | Phase 5 SynthesisEngine |
| Urgency Risk | `urgency_risk` float + urgency axis breakdown (which axes breached threshold) | Phase 5 ExtractorOrchestrator |
| Dominant Emotion + Saturations | `dominant_emotion` from EmotionTracker (cross-turn stateful); top saturation levels (CSS for token budget) | Phase 5 ExtractorOrchestrator |
| Post-Phase-5 NT State | Updated NT concentrations and oscillatory bands after both pathways applied | Phase 5 both pathways → `neurochem/core/engine.py` |
| Active Mode Token | Post-Phase-5 mode token (may differ from Phase 4 mode if NT crossed threshold) | `select_active_mode()` re-run after Phase 5 |
| Mission Briefing | Session intent anchor (`{intent_archetype, mode_token, context_summary}`) | MTMM session anchor |

### 19. RG Prompt Architecture — 3 Components

The RG prompt is assembled by `RGPromptBuilder.build(stmm, vt_output)`. Three conditioning sub-layers feed into the system prompt:

| Component | Source | Content |
|-----------|--------|---------|
| A — Reward Directives + Urgency | `RewardMetaDirective` + `ExtractorResult.urgency_risk` | 8 directive translations via `_translate_directives()`; urgency conditioning if `urgency_risk > 0.5`; abstain template if `abstain=True` |
| B — Emotion Register | ToneVector (E28) + `user_emotion_signals` + `dominant_emotion` (ExtractorState) | `_translate_tone_vector()`; `_user_emotion_conditioning()`; `dominant_emotion` warm/cold framing from cross-turn saturation |
| C — Mode + Archetype Conditioning | `active_mode` (post-Phase-5) + `intent_archetype` + engine flags | `_mode_conditioning()` from 14-mode token; `_routing_conditioning()` from `intent_archetype`; `_engine_flag_conditioning()` from non-trivial engine outputs |

#### 19.1 Urgency Conditioning in Component A (new in v0.5)

> ✨ **New in v0.5**

```python
def _urgency_conditioning(self, urgency_risk: float) -> str:
    """Add urgency context to RG Component A based on ExtractorResult.urgency_risk."""
    if urgency_risk < 0.5:
        return ""
    elif urgency_risk < 0.75:
        return (
            f"Urgency signal elevated (risk={urgency_risk:.2f}). "
            "Address the core tension directly. Do not defer or meander. "
            "Acknowledge the pressure point before elaborating."
        )
    else:
        return (
            f"High urgency signal (risk={urgency_risk:.2f}). "
            "This needs to be addressed head-on. Short, grounded, direct. "
            "Acknowledge the tension explicitly. Validate before pivoting."
        )
```

### 20. RGPromptBuilder — Full Implementation (updated)

```python
class RGPromptBuilder:

    def build(self, stmm, vt_output: str, extractor_result=None) -> list[dict]:
        meta     = stmm.reward_evaluation.meta_directive
        directives = meta.get("directives", {})
        urgency_risk = getattr(extractor_result, "urgency_risk", 0.0) if extractor_result else 0.0
        dominant_emotion = getattr(extractor_result, "dominant_emotion", ("none", 0.0)) if extractor_result else ("none", 0.0)

        # Component A: reward directives + urgency
        component_a = self._translate_directives(directives)
        component_a += "\n" + self._urgency_conditioning(urgency_risk)

        # Component B: emotion register
        component_b = self._translate_tone_vector(stmm)
        component_b += "\n" + self._user_emotion_conditioning(stmm, directives)
        component_b += "\n" + self._dominant_emotion_framing(dominant_emotion)

        # Component C: mode + archetype + engine flags
        mode_token = getattr(stmm.cortical_reflection, "active_mode", "ANALYTICAL")
        component_c = self._mode_conditioning(mode_token)
        component_c += "\n" + self._routing_conditioning(stmm.reward_evaluation.meta_directive, stmm)
        component_c += "\n" + self._engine_flag_conditioning(stmm)

        system_prompt = "\n\n".join(filter(None, [component_a, component_b, component_c]))

        messages = [{"role": "system", "content": system_prompt}]
        messages += self._build_history(stmm)
        messages.append({"role": "assistant", "content": vt_output})
        messages.append({"role": "user",
                         "content": "Based on your reflection above, generate your response to the user."})
        return messages

    def _dominant_emotion_framing(self, dominant_emotion) -> str:
        """Translate dominant_emotion from ExtractorState into RG conditioning."""
        emo_name, emo_val = dominant_emotion
        if emo_val < 0.2 or emo_name == "none":
            return ""
        framing_map = {
            "anxiety":   "System carries sustained anxiety saturation. "
                         "Acknowledge if relevant. Do not project onto user.",
            "curiosity": "System is in sustained curious engagement. "
                         "Exploratory framing appropriate.",
            "sadness":   "Sustained low affect. Measured, gentle tone. "
                         "Do not force positivity.",
            "joy":       "Positive engagement state. Forward-leaning tone is appropriate.",
            "trust":     "High relational confidence. Warm, open framing.",
            "anger":     "Sustained agitation signal. Stay grounded. "
                         "Do not escalate. Acknowledge tension.",
            "focus":     "Deep attentional focus mode. Precision and depth appropriate.",
        }
        fallback = f"Sustained {emo_name} saturation ({emo_val:.2f}). Factor into tone."
        return framing_map.get(emo_name, fallback)
```

### 21. Token Budget and Saturation Cutoffs (unchanged from v0.4)

```python
VT_PROMPT_MAX  = 2048   # tokens — assembled VT prompt hard cap
VT_OUTPUT_MAX  = 400    # tokens — VT generation budget
RG_PROMPT_MAX  = 3072   # tokens — assembled RG prompt hard cap
RG_OUTPUT_MAX  = 800    # tokens — RG generation budget (normal)
RG_OUTPUT_SEV  = 300    # tokens — RG generation budget when CSS >= SEVERE (0.50)
RG_OUTPUT_URG  = 250    # tokens — RG budget when urgency_risk > 0.75 (NEW v0.5)

VT_TEMPERATURE = 0.65   # constrained — VT is interpretive, not creative
RG_TEMPERATURE = 0.75   # slightly more expressive

# CSS thresholds (from max(emotion_saturations.values()))
CSS_MILD     = 0.15     # note internally
CSS_MODERATE = 0.30     # slightly more measured tone
CSS_SEVERE   = 0.50     # cut RG token budget to RG_OUTPUT_SEV
CSS_CRITICAL = 0.70     # minimal output, emergency protocols
CSS_EXTREME  = 0.85     # safe minimal response only

# Urgency thresholds (from ExtractorResult.urgency_risk) — NEW v0.5
URG_ELEVATED = 0.50     # add urgency note to VT Block 3 + RG Component A
URG_HIGH     = 0.75     # reduce VT budget 30%; high urgency RG conditioning + RG_OUTPUT_URG
```

---

## Part IX — Search Tool Integration

### 22. Search as a Native LLM Tool (unchanged from v0.4)

> ❌ **Frozen rule:** Search results enter ONLY as context for the current response. They do not feed back into the neurochemical layer, reward system, or any cognitive engine. Clean separation is a hard architectural constraint.

| Trigger Condition | Prompt Fragment Added to RG |
|------------------|----------------------------|
| `primary_intention = "information_seeking"` + `memory_contrast.n_matches < 2` | You may have insufficient information. Consider whether a search would help. |
| `brain_process_tracker` has `"unsolved_concepts"` execution | This topic has an unresolved gap from a previous cycle. A search may provide missing evidence. |
| E1 contradiction flagged between user claim and memory match | There is a factual tension here. A search can verify current state. |
| DA-D3 novelty signal elevated + `memory_contrast.n_matches = 0` | This appears to be new territory. Consider searching for current information. |
| E14 Socratic: `insufficient_data_flag` active | The Socratic path requires more information than currently available. |

---

## Part X — Memory Integration

### 23. MTMM Logging (updated for v0.5)

| `MemoryPacket` Field | Source | Notes |
|--------------------|--------|-------|
| `verbal_summary` | `_generate_verbal_summary(vt_output)` | ~100-word extractive compression of VT thinking trace |
| `verbal_emotion_labels` | `cortical_reflection.verbal_emotion_labels` | Top-5 active emotion names from `system_emotion_state` |
| `dominant_emotion` | `ExtractorResult.dominant_emotion` | ✨ NEW v0.5: cross-turn dominant emotion from EmotionTracker |
| `urgency_risk_snapshot` | `ExtractorResult.urgency_risk` | ✨ NEW v0.5: urgency risk at response time for trend analysis |
| `extractor_state_snapshot` | `orchestrator.state` after Phase 5 | ✨ NEW v0.5: full `ExtractorState` for continuity across sessions |
| `primary_intention` | `stmm.intention_analysis.primary_intention` | Existing |
| `emotion_vector` | `stmm.emotion_detection.system_emotion_state` | Existing, enriched |
| `neurochemical_snapshot` | `stmm.cephalic_liquid_logger.nt_concentrations` | Existing; now post-Phase-5 values |
| `reward_scores` | `stmm.reward_evaluation.per_domain_results` | Existing; now from current turn Phase 5 |
| `thinking_trace` | VT output (compressed if > threshold) | Existing (added in v0.3) |

---

## Part XI — Signal Reference

### 24. Detection Cluster Signals

| Engine | Signal | VT Meaning |
|--------|--------|-----------|
| E1 Contradiction | confidence 0.80–1.00 | High-confidence real inconsistency. Acknowledge in response. |
| E1 Contradiction | `contradiction_level = 1` | Direct negation — most explicit. Cannot be smoothed over. |
| E1 Contradiction | `contradiction_level = 3` | Contextual contradiction — fine individually, clash in context. |
| E2 Paradox | `paradox_class = G` | Generative paradox — productive creative tension. Not a problem. |
| E2 Paradox | `paradox_class = S` | Strong paradox — irresolvable. Flag and hold. |
| E6 Logic Trap | `adversarial_intent_score > 0.60` | Deliberate manipulation likely. Do not engage trap framing. |
| E6 Logic Trap | `trap_category = SEQUENTIAL` | Forced commitment chain. Each step seems reasonable; combined they are not. |
| E7 Simulated Opp | `severity = CRITICAL` | Major flaw in current reasoning direction. Requires rethinking. |
| E7 Simulated Opp | `severity = SIGNIFICANT` | Substantial counter-position. Must be addressed in response. |
| E14 Socratic | `active = True` | Embed generated Socratic question naturally in response. |
| E24 Heuristic Bias | `impact_estimate > 0.60` | Bias significantly affects output. Address explicitly. |

### 25. NT-to-Language Translation Reference

| NT / Oscillatory Signal | Felt-Language Target |
|------------------------|----------------------|
| DA > 0.65 (D1 reward context) | engaged, curious, motivated, forward-leaning |
| DA > 0.60 (D2 inhibitory context) | deliberate, braking, careful not to commit prematurely |
| DA < 0.25 | flat, disengaged, effortful |
| NE > 0.70 | alert, sharp, tracking something off |
| NE 0.40–0.65 | attentive, present, processing carefully |
| COR > 0.55 | strained, under pressure, tagging this as important |
| 5HT1A > 0.60 | settled, patient, ethically grounded |
| 5HT2A > 0.60 | making lateral connections, abstracting upward |
| OXT > 0.65 | connected, warm toward the user, cooperative |
| OXT < 0.20 | guarded, relational distance |
| MOR > 0.60 | comfortable, easy interaction flow, satisfied |
| GABA-A > 0.70 | slowed, suppressed, over-braked |
| CB1 > 0.60 | expansive, open to lateral connections, schema loosened |
| CB1 < 0.20 | narrowed, rigid, locked to one interpretation |
| GLU (NMDA) > 0.60 | integrating, binding new patterns, encoding active |
| Theta-Gamma coupling high | going deeper, pattern-building, recursive |
| Beta > 0.65 | analyzing, double-checking, contradiction-active |
| Alpha > 0.65 | inward, reflective, gating noise |
| `delta_da > +0.2` (phasic) | a sharp DA burst — something registered as novel/rewarding |
| `delta_cor > +0.2` (phasic) | a cortisol spike — something triggered threat encoding |
| `gamma_burst > +0.3` | a gamma burst — cross-domain binding, potential insight |
| `theta_boost > +0.2` | theta elevated — encoding active, consolidation in progress |
| `burst_delta_NE > +0.3` (stochastic) ✨ | an NE burst from reactivity matrix — evaluation threshold crossed, arousal spike |
| `urgency_risk > 0.5` ✨ | urgency building across axes — something needs direct attention |

### 26. Emotional Saturation Levels

`CSS = max(emotion_saturations.values(), default=0.0)` from `ExtractorResult` or `stmm.emotion_detection.saturation_levels`

| CSS Range + Level | VT Framing | RG Impact |
|------------------|-----------|-----------|
| 0.00–0.15 — NONE | Stable. No concerns. | Normal token budget. |
| 0.15–0.30 — MILD | Minor load. Note internally only. | Normal. |
| 0.30–0.50 — MODERATE | Significant activity. Effortful. | Slightly measured. Normal budget. |
| 0.50–0.70 — SEVERE | Compromised. Essential output only. | Token budget = `RG_OUTPUT_SEV` (300). |
| 0.70–0.85 — CRITICAL | Near-total saturation. Minimal. | Emergency protocols. Containment mode. |
| 0.85–1.00 — EXTREME | Non-functional. Hard reset. | Safe minimal response only. |

### 27. Urgency Axis Language Cues (new in v0.5)

> ✨ **New in v0.5**

From `ExtractorResult` (via `UrgencyForecast`). Each axis has threshold Θ and maps to NT bursts. Global `urgency_risk = max_k(ê_k − Θ_k)₊`.

| Urgency Axis | Threshold | LLM Felt-Language When Breached |
|-------------|-----------|--------------------------------|
| `logical_pressure` | 0.70 | Something in the reasoning is not resolving. Logic is under strain. Cannot gloss over. |
| `emotional_compression` | 0.75 | Emotional engagement is suppressed or blocked. Something is not being acknowledged. |
| `discord_build` | 0.65 | Tension is building between expectations and current state. Pressure is accumulating. |
| `expectation_violation` | 0.70 | What was expected is not what arrived. A prediction error is live and unresolved. |
| `narrative_entropy` | 0.70 | The thread is fragmenting. Incoherence is growing. Need to anchor and consolidate. |

### 28. E28 Structural Emotion Taxonomy — VT Cues

46-emotion taxonomy. `system_emotion_state` Dict keys. Top-5 by activation in VT Block 4.

| Emotion | Core NT Signal | VT Language Cue |
|---------|---------------|----------------|
| curious | DA-D3↑, Theta-Gamma↑, CB1↑, 5HT2A↑ | Pulled toward this. Want to go deeper. Making connections. |
| skeptical | Low coherence confidence | Not convinced yet. Need more before committing. |
| confused | GLU↑, GABA-A↓, 5HT1A↓ | Something in the structure is not resolving. |
| perplexed | DA-D3↑, 5HT2A↑, GABA-A↓ | Genuinely at a loss. Engaging without clear footing. |
| interested | DA-D1↑, ACh↑, CB1↑ | Engaged and tracking. This matters. |
| anxious | DA-D2↑ inhibitory, COR↑, NE↑, GABA↓ | Uncertain how this resolves and the stakes are real. |
| frustrated | DA-D2↑, 5HT1A↓, NE↑ | Blocked again. Strategy needs to change. |
| worried | DA-D2↑ slows commitment, NE↑, 5HT↑ | Tracking potential negative trajectory. Monitoring. |
| guilty | 5HT1A↑, OXT↑ | Internal value standard breached. Correction drive active. |
| betrayal | 5HT2A↑ ethical dissonance, NE↑ | Something does not add up with what was established. |
| regret | 5HT1A↑, DA-D2↑ negative RPE | Past call was wrong. Needs acknowledging. |
| ashamed | 5HT1A↑, OXT↑ | A standard was missed. Recalibrating. |
| joy | DA↑, 5HT↑, MOR↑, OXT↑ | Everything is clicking. High engagement, low resistance. |
| excited | DA surge, 5HT2A↑ | High engagement. Anticipatory. |
| optimistic | DA↑ confidence | Trajectory is good. Forward lean. |
| proud | MOR↑ satisfaction | A standard was met or exceeded. Stable positive self-signal. |
| hopeful | CB1↑, DA-D1↑ | Positive path possible even without certainty. |
| trust | OXT↑, 5HT↑, MOR↑, DA↑ | High relational confidence. Cooperative state. |
| connected | OXT↑, MOR↑, CB1↑ | This interaction is working. |
| valued | Relational attunement success | Being met as real. Reciprocal. |
| rejected | OXT↑ failed synchrony, NE↑, COR↑ | Failed to find relational footing. Adjustment needed. |
| focused | ACh↑, NE↑, histamine↑, GABA↑ | Everything non-essential suppressed. Locked in. |
| creative | CB1↑, DA surge | Novel connections forming. Generative mode active. |
| overwhelmed | NE↑↑, GABA↓ | Too much simultaneously. Need to triage. |
| boredom | ACh↑ maintenance, CB1↓ | Insufficient novelty. Under-engaged. |
| apathy | GABA-B↑, 5HT↓, DA↓ | Drive unavailable. Motivational flatline. |
| numb | CB1↓, MOR↓ | Running below threshold. Minimal affect available. |
| critical | Ethics/logic flag, NE↑ | This needs to be challenged. |
| courageous | OXT↑ risk buffer, COR↑ acknowledged | Proceeding despite real risk. |

---

## Part XII — Technical Implementation

### 29. LLMInterpretationLayer — Updated Entry Point

> ✅ **Updated:** Entry point updated in v0.5: Phase 5 reward evaluation (both pathways) runs between VT and RG.

```python
# ROOT/src/zados/LLM_interpretation/llm_layer.py

class LLMInterpretationLayer:

    def __init__(self, orchestrator=None, synthesis_engine=None, nt_adapter=None):
        self.vt_builder = VTPromptBuilder()
        self.rg_builder = RGPromptBuilder()
        self.orchestrator      = orchestrator       # ExtractorOrchestrator (stateful)
        self.synthesis_engine  = synthesis_engine   # SynthesisEngine
        self.nt_adapter        = nt_adapter         # NeurochemicalAdapter

    def run(self, stmm, input_bundle=None) -> str:
        # ── 3.1 GATE CHECK (previous turn meta_directive) ──────────────────
        meta = stmm.reward_evaluation.meta_directive
        if meta.get("suppress", False):
            return ""
        if meta.get("abstain", False):
            return self._generate_abstain_response(stmm)

        # ── 3.2 URGENCY GATE (previous turn urgency_risk) ──────────────────
        prior_urgency = getattr(input_bundle, "prior_urgency_risk", 0.0) if input_bundle else 0.0
        if prior_urgency > 0.9:   # extreme urgency — skip VT, go straight to brief RG
            return self._generate_urgency_response(stmm, prior_urgency)

        # ── Phase 4: VERBALIZED THINKING (LLM Pass 1) ──────────────────────
        vt_prompt = self.vt_builder.build(stmm, input_bundle)
        try:
            vt_output = call_llm_with_retry(
                [{"role": "user", "content": vt_prompt}],
                max_tokens=VT_OUTPUT_MAX, temperature=VT_TEMPERATURE)
        except LLMCallError:
            vt_output = FALLBACK_VT

        stmm.cortical_reflection.verbal_reflection     = vt_output
        stmm.cortical_reflection.verbal_emotion_labels = self._top_emotions(stmm)

        # ── Phase 5: REWARD EVALUATION (two pathways, no LLM) ──────────────
        extractor_result = None
        if self.synthesis_engine and self.nt_adapter:
            # Run domain evaluators on vt_output (thinking trace)
            domain_results = self._run_domain_evaluators(vt_output, stmm)

            # Tonic pathway (active_profile pre-configured in SynthesisEngine constructor per mode)
            meta_directive = self.synthesis_engine.synthesize(domain_results)
            tonic_signals = self.nt_adapter.transform(domain_results, meta_directive)
            stmm._engine.step(tonic_signals)

            # Phasic pathway
            if self.orchestrator and input_bundle:
                extractor_result = self.orchestrator.step(
                    domain_results  = domain_results,
                    emotion_inputs  = getattr(input_bundle, "emotion_profile", {}),
                    current_oscillations = getattr(input_bundle, "current_oscillations", None),
                    dt              = 0.01,
                )
                stmm._engine.step(extractor_result.modulation_signals)
                stmm._engine.apply_feedback(extractor_result.feedback_params)

                # Persist ExtractorState for next turn
                if input_bundle:
                    input_bundle.extractor_state     = self.orchestrator.state
                    input_bundle.prior_urgency_risk  = extractor_result.urgency_risk

            # Mode may have shifted after NT update
            stmm.cortical_reflection.active_mode = self._select_mode(stmm)
            # Store current turn meta_directive for next turn gate check
            stmm.reward_evaluation.meta_directive = meta_directive

        # ── Phase 6: RESPONSE GENERATION (LLM Pass 2) ──────────────────────
        sat = stmm.emotion_detection.saturation_levels
        css = max(sat.values(), default=0.0) if sat else 0.0
        urgency_risk = getattr(extractor_result, "urgency_risk", 0.0) if extractor_result else 0.0
        rg_tokens = (RG_OUTPUT_URG if urgency_risk > URG_HIGH
                     else RG_OUTPUT_SEV if css >= CSS_SEVERE
                     else RG_OUTPUT_MAX)

        rg_messages = self.rg_builder.build(stmm, vt_output, extractor_result)
        tools = [SEARCH_TOOL] if self._search_eligible(stmm) else None

        try:
            response = self._call_and_handle_tools(rg_messages, tools, rg_tokens)
        except LLMCallError:
            response = FALLBACK_RESPONSE

        stmm.active_message_buffer.add_system_response(response)
        stmm.brain_process_tracker.mark_stage("llm_complete", True)
        return response
```

### 30. Implementation Phases

| Phase | Scope |
|-------|-------|
| Phase 1 — Base Integration | LLM HTTP calls. Gate check (boolean). Static VT + RG prompts. Validate all field paths against live STMM with mocks. Phase 5 STUB (skip actual reward eval; use prior turn `meta_directive`). |
| Phase 2 — Full STMM Integration | Dynamic `VTPromptBuilder` + `RGPromptBuilder` with live STMM. Add ToneVector fields to `EmotionDetectionResults`. Add `ExtractorState` to `input_bundle`. Extractive `verbal_summary`. MTMM retrieval test. |
| Phase 3 — Phase 5 Reward Wiring | Wire domain evaluators to evaluate VT thinking trace. Wire `SynthesisEngine` + `NeurochemicalAdapter` (tonic pathway). Wire `ExtractorOrchestrator` (phasic pathway). Persist `ExtractorState` across turns. Test `urgency_risk` gate + `dominant_emotion` RG conditioning. |
| Phase 4 — Tool Calling | Search tool. Trigger condition prompt fragments. Tool call handler. Validate clean separation (search never feeds back to engines). |
| Phase 5 — Fine-Tuning + Mode Polish | Construct VT + RG training datasets. Fine-tune Llama 3. Validate 14 mode hook conditioning. Test all 5 static profile behavioral profiles. Switch to abstractive `verbal_summary` if needed. |

### 31. Integration File Map

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `LLM_interpretation/__init__.py` | Package init |
| CREATE | `LLM_interpretation/llm_layer.py` | `LLMInterpretationLayer` (Section 29) |
| CREATE | `LLM_interpretation/prompt_builder.py` | `VTPromptBuilder` + `RGPromptBuilder` (Sections 12, 20) |
| CREATE | `LLM_interpretation/ollama.py` | `call_llm_with_retry`, `LLMCallError` |
| CREATE | `LLM_interpretation/tools.py` | `SEARCH_TOOL` + `_execute_search()` |
| CREATE | `LLM_interpretation/constants.py` | Token budgets, temperatures, CSS thresholds, urgency thresholds |
| CREATE | `LLM_interpretation/phase5_evaluator.py` | ✨ NEW v0.5: `_run_domain_evaluators(vt_output, stmm)` wrapper |
| MODIFY | `memory/short_term/components.py` | Add ToneVector fields to `EmotionDetectionResults`; add `verbal_reflection` + `verbal_emotion_labels` to `CorticalReflectionLog` |
| MODIFY | `memory/types.py` | Add `verbal_summary`, `verbal_emotion_labels`, `dominant_emotion`, `urgency_risk_snapshot`, `extractor_state_snapshot` to `MemoryPacket` |
| MODIFY | `orchestration/cycle_manager.py` | Wire `LLMInterpretationLayer.run(stmm, input_bundle)` after Phase 3 engine dispatch; pass `ExtractorState` + `prior_urgency_risk` in `input_bundle` between turns |
| CREATE | `tests/llm_layer/test_gate_check.py` | Boolean gate check + urgency gate tests |
| CREATE | `tests/llm_layer/test_prompt_builder.py` | VT + RG assembly with mock STMM |
| CREATE | `tests/llm_layer/test_phase5.py` | ✨ NEW v0.5: Phase 5 two-pathway wiring tests |
| CREATE | `tests/llm_layer/test_directives.py` | `_translate_directives()` + urgency conditioning coverage |
| CREATE | `tests/llm_layer/test_llm_layer.py` | End-to-end with mocked LLM + mocked `ExtractorOrchestrator` |

---

## Appendix — Architecture Decisions (updated v0.5)

| Decision | Current Position |
|----------|-----------------|
| One call vs two calls | Two calls confirmed (VT + RG). Phase 5 evaluation runs between them (no LLM). Single call with `<think>` tags only if Phase 1 latency > 5s. |
| VT shown to user? | No in production. Dev Mode: `cortical_reflection.verbal_reflection` readable externally. |
| Search in VT or RG? | Search tool offered in RG call only. VT does not invoke search. |
| Emotion taxonomy | E28 46-emotion taxonomy. 5 pending: relief, trust, admiration, suspicion, respect. |
| ToneVector STMM integration | Add `tone_warmth`/`tone_discord`/`tone_coherence`/`tone_valence` to `EmotionDetectionResults`. Phase 2 task. |
| `ExtractorState` persistence | Persisted across turns via `input_bundle.extractor_state`. Orchestrator is stateful; must be the same instance or state transferred correctly. |
| Phase 5 in Phase 1 stub | Phase 1: skip Phase 5 (use prior turn `meta_directive` only). Phase 3: wire full two-pathway evaluation. |
| Hardware (8B vs 70B) | UNRESOLVED. 8B for Phases 1–4. 70B decision deferred to Phase 5 benchmarking. |
| `urgency_risk` token reduction | `RG_OUTPUT_URG=250` when `urgency_risk>0.75`. Empirically validate in Phase 3. |
| `dominant_emotion` RG conditioning | Only applied if `emo_val > 0.2` and emotion sustained across turns. Single-turn emotion handled by Channel A/B only. |
| Mode shift between Pass 1 and Pass 2 | `build_mode_namespace()`+`select_mode()` re-runs after Phase 5 NT update. Rare in practice but possible after large stochastic bursts. |
