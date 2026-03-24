"""
ZA-DOS LLM Interpretation Layer — prompt builders (v0.5).

VTPromptBuilder  — assembles the 5-block Verbalized Thinking prompt.
RGPromptBuilder  — assembles the OpenAI-format messages list for Response Generation.

v0.5 changes from v0.3
----------------------
VT:
  - build() now accepts input_bundle for extractor state + mission briefing
  - Block 1 extended with reward profile name + mission briefing
  - Block 4 extended with ExtractorState snapshot + urgency framing
  - Block 5 adds reward evaluation quality warning when composite < 0.3

RG:
  - build() now accepts extractor_result + input_bundle + selected_mode
  - System message restructured into Components A / B / C:
      A — Directives (asymmetric thresholds) + mode conditioning + urgency
      B — Emotion framing + ToneVector + user distress
      C — Engine flags + memory contrast
  - VT moved from system message to assistant message (not shown to user)
  - Conversation history: user messages only; system responses as assistant
  - New methods: _mode_conditioning, _urgency_conditioning,
    _dominant_emotion_framing, _format_extractor_emotions

Neither class makes any LLM calls; both are pure data assemblers.
All STMM field access is read-only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from zados.LLM_interpretation.constants import (
    ARCHETYPE_CONDITIONING,
    CSS_MODERATE,
    CSS_SEVERE,
    DIRECTIVE_THRESHOLDS,
    EMOTION_FRAMING,
    FLAG_KEYWORDS,
    MODE_CONDITIONING,
    URG_ELEVATED,
    URG_HIGH,
    VT_PROMPT_MAX,
)


# ---------------------------------------------------------------------------
# Helper: approximate token count  (4 chars ≈ 1 token — coarse estimate)
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ============================================================================
# VTPromptBuilder  (v0.5)
# ============================================================================

class VTPromptBuilder:
    """
    Assembles the 5-block Verbalized Thinking prompt.

    The assembled text is passed to the LLM as a single user message
    asking for an internal monologue — NOT a user-facing response.

    Block layout (v0.5):
        1  Identity, mode, profile, mission briefing
        2  User input summary
        3  Cognitive engine findings (non-trivial only)
        4  Neurochemical & emotional state + ExtractorState + urgency
        5  VT generation instruction + reward eval quality warning
    """

    def build(
        self,
        stmm,
        input_bundle: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Assemble all 5 blocks.

        Parameters
        ----------
        stmm : STMMStore
            Current short-term memory state.
        input_bundle : dict, optional
            Pipeline context carrying extractor_state, mission_briefing,
            active_reward_profile_name, prior_urgency_risk, etc.

        If the estimated token count exceeds VT_PROMPT_MAX, Block 3 is
        truncated to the top-3 flagged engines before retrying.
        """
        bundle = input_bundle or {}

        b1 = self._block1(stmm, bundle)
        b2 = self._block2(stmm)
        b4 = self._block4(stmm, bundle)
        b5 = self._block5(stmm, bundle)
        b3_full = self._block3(stmm)

        full = "\n\n".join([b1, b2, b3_full, b4, b5])
        if _approx_tokens(full) > VT_PROMPT_MAX:
            b3_short = self._truncate(stmm)
            full = "\n\n".join([b1, b2, b3_short, b4, b5])

        return full

    # ------------------------------------------------------------------
    # Block 1 — Identity, Mode, Profile, Mission
    # ------------------------------------------------------------------

    def _block1(self, stmm, bundle: Dict[str, Any]) -> str:
        cr = stmm.cortical_reflection
        anomalies_str = (
            ", ".join(cr.processing_anomalies)
            if cr.processing_anomalies
            else "none"
        )

        lines = [
            "You are ZA-DOS's internal cognitive voice. "
            "You are NOT generating a response to the user.",
            "You are translating the system's internal computational state "
            "into a natural-language monologue.",
            "",
            f"Mode: {cr.active_mode} | "
            f"Cycle: {stmm._turn_index} | "
            f"Identity coherence: {cr.identity_coherence_status}",
            f"Anomalies: {anomalies_str}",
        ]

        # v0.5: active reward profile
        profile_name = bundle.get("active_reward_profile_name", "")
        if profile_name:
            lines.append(f"Active reward profile: {profile_name}")

        # v0.5: mission briefing
        mission = bundle.get("mission_briefing", "")
        if mission:
            lines.append(f"Mission: {mission}")

        # v0.6: operational context flags — pipeline origin + active mode overrides
        ctx_flags = bundle.get("context_flags", {})
        if ctx_flags:
            # Pipeline origin tag (pipeline:*)
            pipeline_tag = next(
                (v for v in ctx_flags if isinstance(v, str) and v.startswith("pipeline:")),
                None,
            )
            # Also check key-based flags set by learning/sleep pipelines
            if pipeline_tag is None:
                for k in ctx_flags:
                    if isinstance(k, str) and k.startswith("pipeline:"):
                        pipeline_tag = k
                        break
            if pipeline_tag:
                lines.append(f"Pipeline context: {pipeline_tag.split(':', 1)[1]}")
            # Notable mode flags
            op_active = [
                k for k in ctx_flags
                if isinstance(k, str) and k in (
                    "dream_mode", "autonomous_mode", "e28_disabled",
                    "retroactive_contrast", "learning_reframe", "confusion_override",
                )
            ]
            if op_active:
                lines.append(f"Active context overrides: {', '.join(op_active)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Block 2 — User Input Summary
    # ------------------------------------------------------------------

    def _block2(self, stmm) -> str:
        ia     = stmm.intention_analysis
        latest = stmm.active_message_buffer.latest_user()
        user_text = latest.text if latest else "(no input)"
        sub_str   = (
            ", ".join(ia.sub_intentions) if ia.sub_intentions else "none"
        )
        return (
            "USER INPUT SUMMARY:\n"
            f"Input: {user_text}\n"
            f"Primary intention: {ia.primary_intention} "
            f"(confidence: {ia.confidence:.2f})\n"
            f"Secondary: {sub_str}\n"
            f"Pressure type: {ia.pressure_type} | "
            f"Stability: {ia.stability_passed}"
        )

    # ------------------------------------------------------------------
    # Block 3 — Cognitive Engine Findings (non-trivial only)
    # ------------------------------------------------------------------

    def _block3(self, stmm) -> str:
        """
        Include only engine executions whose output_summary contains at
        least one FLAG_KEYWORDS term.  Always appends memory-contrast
        stats and reward domain scores.
        """
        lines = ["COGNITIVE ENGINE FINDINGS (non-trivial only):"]

        for ex in stmm.brain_process_tracker.executions:
            if ex.skipped:
                continue
            if any(kw in ex.output_summary.lower() for kw in FLAG_KEYWORDS):
                lines.append(f"  {ex.engine_id}: {ex.output_summary}")

        mc = stmm.memory_contrast
        lines.append(
            f"Memory Contrast: {len(mc.matched_entries)} matches | "
            f"{len(mc.potential_contradictions)} contradictions | "
            f"{len(mc.unresolved_query_matches)} unresolved"
        )

        re       = stmm.reward_evaluation
        meta     = re.meta_directive or {}
        meta_sub = meta.get("meta", {}) or {}
        per_dom  = meta_sub.get("per_domain_weighted_scores", {}) or {}
        eth  = per_dom.get("ethics",           0.0)
        log  = per_dom.get("logic",            0.0)
        inn  = per_dom.get("innovation",       0.0)
        att  = per_dom.get("human_attunement", 0.0)
        tier = meta_sub.get("composite_tier_label", "—")
        lines.append(
            f"Reward: Ethics {eth:.2f} | Logic {log:.2f} | "
            f"Innovation {inn:.2f} | Attunement {att:.2f} | "
            f"Composite {re.composite_score:.2f} | Tier: {tier}"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Block 4 — Neurochemical & Emotional State + ExtractorState + Urgency
    # ------------------------------------------------------------------

    def _block4(self, stmm, bundle: Dict[str, Any]) -> str:
        cll = stmm.cephalic_liquid_logger
        nt  = cll.nt_concentrations
        osc = cll.oscillatory_bands
        ed  = stmm.emotion_detection

        # NT snapshot
        nt_line = (
            f"NT: DA={nt.get('da', 0.0):.3f}  NE={nt.get('ne', 0.0):.3f}  "
            f"5HT={nt.get('5ht', 0.0):.3f}  ACh={nt.get('ach', 0.0):.3f}\n"
            f"    GLU={nt.get('glu', 0.0):.3f}  GABA={nt.get('gaba', 0.0):.3f}  "
            f"COR={nt.get('cor', 0.0):.3f}  OXT={nt.get('oxt', 0.0):.3f}  "
            f"MOR={nt.get('mor', 0.0):.3f}  CB1={nt.get('cb1', 0.0):.3f}"
        )

        # Oscillatory bands
        osc_line = (
            f"Osc: D={osc.get('delta', 0.0):.3f}  Th={osc.get('theta', 0.0):.3f}  "
            f"Al={osc.get('alpha', 0.0):.3f}  Be={osc.get('beta', 0.0):.3f}  "
            f"Ga={osc.get('gamma', 0.0):.3f}"
        )

        # CSS
        sat      = ed.saturation_levels
        css      = max(sat.values(), default=0.0) if sat else 0.0
        sat_type = max(sat, key=sat.get) if sat else "none"
        sat_line = f"Saturation: CSS={css:.3f} | Dominant: {sat_type}"

        # Emotions
        sys_em  = self._fmt_system_emotions(ed)
        user_em = self._fmt_user_emotions(ed)
        em_line = (
            f"System emotions: {sys_em}\n"
            f"User emotions:   {user_em}"
        )

        # ToneVector
        tv_line = (
            f"ToneVector: valence={ed.tone_valence:.2f}  "
            f"coherence={ed.tone_coherence:.2f}  "
            f"warmth={ed.tone_warmth:.2f}  "
            f"discord={ed.tone_discord:.2f}"
        )

        # Phasic deltas
        phasic_line = self._fmt_phasic_deltas(stmm)

        sections = [
            "INTERNAL STATE:",
            nt_line,
            osc_line,
            sat_line,
            em_line,
            tv_line,
            phasic_line,
        ]

        # v0.5: ExtractorState snapshot from input_bundle
        extractor_state = bundle.get("extractor_state")
        if extractor_state is not None:
            ext_line = self._fmt_extractor_state(extractor_state)
            if ext_line:
                sections.append(ext_line)

        # v0.5: Urgency framing
        prior_urgency = bundle.get("prior_urgency_risk", 0.0)
        if prior_urgency >= URG_ELEVATED:
            urgency_line = self._urgency_framing(prior_urgency)
            sections.append(urgency_line)

        return "\n".join(sections)

    def _fmt_system_emotions(self, ed) -> str:
        state = ed.system_emotion_state
        if not state:
            return "none"
        top5 = sorted(state.items(), key=lambda x: x[1], reverse=True)[:5]
        return ", ".join(f"{k}({v:.2f})" for k, v in top5)

    def _fmt_user_emotions(self, ed) -> str:
        signals = ed.user_emotion_signals
        if not signals:
            return "none"
        top3 = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]
        return ", ".join(f"{k}({v:.2f})" for k, v in top3)

    def _fmt_phasic_deltas(self, stmm) -> str:
        delta = getattr(stmm.emotion_detection, "emotional_delta", {}) or {}

        def _get(key: str, alt: str) -> float:
            return delta.get(key, delta.get(alt, 0.0))

        da    = _get("da",    "delta_da")
        ne    = _get("ne",    "delta_ne")
        oxt   = _get("oxt",   "delta_oxt")
        mor   = _get("mor",   "delta_mor")
        cor   = _get("cor",   "delta_cor")
        ach   = _get("ach",   "delta_ach")
        theta = _get("theta", "theta_boost")
        gamma = _get("gamma", "gamma_burst")

        return (
            "Phasic NT deltas this cycle:\n"
            f"  DA: {da:+.3f} | NE: {ne:+.3f} | OXT: {oxt:+.3f} | "
            f"MOR: {mor:+.3f} | COR: {cor:+.3f} | ACh: {ach:+.3f}\n"
            f"Oscillatory: Theta_boost: {theta:+.3f} | Gamma_burst: {gamma:+.3f}"
        )

    def _fmt_extractor_state(self, extractor_state) -> str:
        """
        Format the ExtractorState for inclusion in VT Block 4.
        Handles both dict and dataclass objects.
        """
        if extractor_state is None:
            return ""

        # Convert to dict if it's a dataclass
        if hasattr(extractor_state, "as_dict"):
            state_dict = extractor_state.as_dict()
        elif isinstance(extractor_state, dict):
            state_dict = extractor_state
        else:
            return ""

        prev_eval = state_dict.get("prev_evaluation_vector")
        if not prev_eval:
            return ""

        # Format the evaluation vector axes
        axes = []
        for axis, val in sorted(prev_eval.items()):
            axes.append(f"{axis}={val:.2f}")

        return "ExtractorState (prev E(t)): " + "  ".join(axes)

    def _urgency_framing(self, urgency: float) -> str:
        """Urgency note for VT Block 4 when urgency >= URG_ELEVATED."""
        if urgency >= URG_HIGH:
            return (
                f"HIGH URGENCY ({urgency:.2f}). "
                "Situation requires immediate, focused response. "
                "Minimise reflection depth. Orient toward action."
            )
        return (
            f"Urgency elevated ({urgency:.2f}). "
            "Note the time-pressure in your reflection. "
            "Be concise but thorough where it matters."
        )

    # ------------------------------------------------------------------
    # Block 5 — VT generation instruction + reward eval quality warning
    # ------------------------------------------------------------------

    def _block5(self, stmm, bundle: Dict[str, Any]) -> str:
        lines = [
            "TASK:",
            "Generate an internal monologue (150-300 words). "
            "First person, present tense. No bullets. No formatting.",
            "Do not address the user.",
            "Reflect cognitive findings, translate NTs into felt-language, "
            "note phasic shifts, flag saturation if CSS > 0.30, "
            "end with how the system is orienting toward the response.",
            "Output ONLY the monologue text.",
        ]

        # v0.5: Reward eval quality warning
        composite = stmm.reward_evaluation.composite_score
        if composite < 0.3:
            lines.append(
                "\nNOTE: Reward composite is low ({:.2f}). "
                "The system's confidence in its own processing quality is "
                "below threshold. Acknowledge this uncertainty in your "
                "reflection.".format(composite)
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Truncation fallback — Block 3 top-3 engines only
    # ------------------------------------------------------------------

    def _truncate(self, stmm) -> str:
        """Block 3 reduced to at most 3 flagged engine entries."""
        lines = ["COGNITIVE ENGINE FINDINGS (truncated - top-3):"]
        count = 0
        for ex in stmm.brain_process_tracker.executions:
            if ex.skipped or count >= 3:
                continue
            if any(kw in ex.output_summary.lower() for kw in FLAG_KEYWORDS):
                lines.append(f"  {ex.engine_id}: {ex.output_summary}")
                count += 1

        mc = stmm.memory_contrast
        lines.append(
            f"Memory Contrast: {len(mc.matched_entries)} matches | "
            f"{len(mc.potential_contradictions)} contradictions"
        )
        return "\n".join(lines)


# ============================================================================
# RGPromptBuilder  (v0.5)
# ============================================================================

class RGPromptBuilder:
    """
    Assembles the OpenAI-format messages list for Response Generation.

    v0.5 structure:
        messages = [
            {"role": "system",    "content": <Component A + B + C>},
            {"role": "assistant", "content": <VT context injection>},
            *history_msgs,
        ]

    System message Components:
        A — Directives (asymmetric thresholds) + mode conditioning + urgency
        B — Emotion framing + ToneVector + user distress conditioning
        C — Engine flags + memory contrast summary

    VT is now an **assistant** message (not part of system), marked as
    internal processing context that should not be repeated to the user.
    """

    def build(
        self,
        stmm,
        vt_output: str,
        extractor_result=None,
        input_bundle: Optional[Dict[str, Any]] = None,
        selected_mode: str = "",
    ) -> List[Dict[str, str]]:
        """
        Assemble the full RG messages list.

        Parameters
        ----------
        stmm : STMMStore
            Current short-term memory state.
        vt_output : str
            VT monologue from Phase 4.
        extractor_result : ExtractorResult, optional
            Phase 5 phasic pathway result (for urgency + emotion sats).
        input_bundle : dict, optional
            Pipeline context.
        selected_mode : str
            Mode token selected after Phase 5 (from MODE_CONDITIONING keys).
        """
        bundle = input_bundle or {}

        system_content = self._build_system(
            stmm, extractor_result, bundle, selected_mode,
        )
        vt_assistant   = self._build_vt_assistant(vt_output)
        history        = self._build_history(stmm)

        messages = [{"role": "system", "content": system_content}]

        # v0.5: VT as assistant message (between system and history)
        if vt_assistant:
            messages.append({"role": "assistant", "content": vt_assistant})

        messages.extend(history)
        return messages

    # ------------------------------------------------------------------
    # VT assistant message (v0.5 — moved out of system message)
    # ------------------------------------------------------------------

    def _build_vt_assistant(self, vt_output: str) -> str:
        """
        Format VT output as an assistant message.
        Returns empty string if no VT output (will be omitted from messages).
        """
        if not vt_output or not vt_output.strip():
            return ""

        return (
            "INTERNAL PROCESSING SUMMARY (do not repeat to user):\n"
            + vt_output
            + "\n\nUse this to calibrate tone, depth, and what to acknowledge. "
            "Let it inform how you speak, not what you say explicitly."
        )

    # ------------------------------------------------------------------
    # System message assembly — Components A / B / C
    # ------------------------------------------------------------------

    def _build_system(
        self,
        stmm,
        extractor_result,
        bundle: Dict[str, Any],
        selected_mode: str,
    ) -> str:
        meta       = stmm.reward_evaluation.meta_directive or {}
        directives = meta.get("directives", {}) or {}

        # Component A: Directives + Mode + Urgency
        a_directives = self._translate_directives(directives)
        a_mode       = self._mode_conditioning(selected_mode, stmm)
        a_urgency    = self._urgency_conditioning(extractor_result, bundle)

        # Component B: Emotion framing + ToneVector + User distress
        b_emotion = self._dominant_emotion_framing(stmm, extractor_result)
        b_tone    = self._tone_conditioning(stmm)
        b_user    = self._user_emotion_conditioning(stmm, directives)

        # Component C: Engine flags + Memory contrast
        c_flags   = self._engine_flag_conditioning(stmm)
        c_memory  = self._memory_contrast_summary(stmm)

        # Component D: Operational context flags (dream, learning, emphasis, etc.)
        d_context = self._context_flag_conditioning(bundle)

        parts = [
            p for p in [
                a_directives, a_mode, a_urgency,
                b_emotion, b_tone, b_user,
                c_flags, c_memory,
                d_context,
            ]
            if p and p.strip()
        ]
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Component A — Directive translations (v0.5 asymmetric thresholds)
    # ------------------------------------------------------------------

    def _translate_directives(self, d: Dict[str, float]) -> str:
        """
        Convert reward directive floats to conditioning prose.

        v0.5: Uses DIRECTIVE_THRESHOLDS with per-directive asymmetric
        thresholds.  A directive is included only if its value exceeds
        its specific threshold.
        """
        if not d:
            return ""

        lines = []
        for key, (threshold, text) in DIRECTIVE_THRESHOLDS.items():
            val = d.get(key, 0.0)
            if val > threshold:
                lines.append(text)

        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # Component A — Mode conditioning (v0.5 — 14 mode tokens)
    # ------------------------------------------------------------------

    def _mode_conditioning(self, selected_mode: str, stmm) -> str:
        """
        Return conditioning prose for the active mode token.

        Checks MODE_CONDITIONING (14 modes) first, then falls back to
        ARCHETYPE_CONDITIONING (8 archetypes).
        """
        if selected_mode and selected_mode in MODE_CONDITIONING:
            return (
                f"Response mode: {selected_mode}\n"
                + MODE_CONDITIONING[selected_mode]
            )

        # Archetype fallback
        archetype = getattr(
            stmm.intention_analysis, "primary_archetype", ""
        ).upper().strip()

        if archetype and archetype in ARCHETYPE_CONDITIONING:
            return (
                f"Response archetype: {archetype}\n"
                + ARCHETYPE_CONDITIONING[archetype]
            )

        return ""

    # ------------------------------------------------------------------
    # Component A — Urgency conditioning (v0.5)
    # ------------------------------------------------------------------

    def _urgency_conditioning(
        self,
        extractor_result,
        bundle: Dict[str, Any],
    ) -> str:
        """
        Add urgency conditioning when urgency_risk is elevated.
        """
        urgency = 0.0

        # Prefer extractor_result urgency, fall back to bundle
        if extractor_result is not None:
            urgency = getattr(extractor_result, "urgency_risk", 0.0)
        if urgency == 0.0:
            urgency = bundle.get("prior_urgency_risk", 0.0)

        if urgency >= URG_HIGH:
            return (
                f"HIGH URGENCY ({urgency:.2f}). "
                "Respond with maximum directness. "
                "Skip elaboration. One clear action or answer. "
                "Minimal token count."
            )
        elif urgency >= URG_ELEVATED:
            return (
                f"Urgency elevated ({urgency:.2f}). "
                "Be concise. Prioritise actionable content. "
                "Reduce hedging."
            )

        return ""

    # ------------------------------------------------------------------
    # Component B — Dominant emotion framing (v0.5)
    # ------------------------------------------------------------------

    def _dominant_emotion_framing(self, stmm, extractor_result) -> str:
        """
        Use EMOTION_FRAMING map for the dominant system emotion.
        Prefers extractor_result.dominant_emotion if available,
        falls back to top system_emotion_state.
        """
        dominant_name = ""
        dominant_val  = 0.0

        # From extractor result
        if extractor_result is not None:
            dom = getattr(extractor_result, "dominant_emotion", None)
            if dom and isinstance(dom, tuple) and len(dom) == 2:
                dominant_name, dominant_val = dom

        # Fallback: top system emotion
        if not dominant_name or dominant_name == "none":
            state = stmm.emotion_detection.system_emotion_state or {}
            if state:
                top = max(state.items(), key=lambda x: x[1])
                dominant_name, dominant_val = top

        if not dominant_name or dominant_val < 0.15:
            return ""

        # Check EMOTION_FRAMING map
        framing = EMOTION_FRAMING.get(dominant_name, "")
        if framing:
            return framing

        # Generic fallback for emotions not in the map
        if dominant_val > 0.4:
            return (
                f"Dominant emotional state: {dominant_name} ({dominant_val:.2f}). "
                "Calibrate your response tone accordingly."
            )

        return ""

    # ------------------------------------------------------------------
    # Component B — ToneVector conditioning
    # ------------------------------------------------------------------

    def _tone_conditioning(self, stmm) -> str:
        ed = stmm.emotion_detection
        warmth    = getattr(ed, "tone_warmth",    0.0)
        discord   = getattr(ed, "tone_discord",   0.0)
        coherence = getattr(ed, "tone_coherence", 0.5)
        valence   = getattr(ed, "tone_valence",   0.0)

        lines = []
        if warmth > 0.4:
            lines.append(
                "System emotional state: warm toward user. "
                "Relational tone is appropriate."
            )
        elif warmth < -0.3:
            lines.append(
                "System emotional state: guarded. Maintain precision. "
                "Do not project warmth you do not have."
            )
        if discord > 0.5:
            lines.append(
                "Internal emotional discord is high. "
                "Acknowledge genuine complexity. "
                "Do not project false certainty."
            )
        if coherence < 0.3:
            lines.append(
                "Emotional coherence is low — mixed signals internally. "
                "Be more cautious in asserting a single stance."
            )
        if valence < -0.4:
            lines.append(
                "Overall valence is negative. Slow down. "
                "Acknowledgment before content. Match user's energy."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Component B — User emotion conditioning (soothe boost on distress)
    # ------------------------------------------------------------------

    _DISTRESS_KEYS = frozenset({
        "anxious", "worried", "nervous", "scared", "afraid",
        "frustrated", "angry", "sad", "rejected", "ashamed",
        "guilty", "overwhelmed", "hurt", "confused", "lost",
    })

    def _user_emotion_conditioning(
        self,
        stmm,
        directives: Dict[str, float],
    ) -> str:
        ed      = stmm.emotion_detection
        signals = ed.user_emotion_signals
        if not signals:
            return ""

        distress_score = sum(
            v for k, v in signals.items()
            if k.lower() in self._DISTRESS_KEYS
        )

        if distress_score > 0.4:
            return (
                "User emotional state shows distress signals. "
                "Prioritise acknowledgment and safety before content. "
                "Soothe conditioning elevated."
            )

        top_user = max(signals, key=signals.get)
        top_val  = signals[top_user]
        if top_val > 0.3:
            return (
                f"User's dominant emotional signal: {top_user} ({top_val:.2f}). "
                "Calibrate accordingly."
            )

        return ""

    # ------------------------------------------------------------------
    # Component C — Engine-specific flag conditioning
    # ------------------------------------------------------------------

    _ENGINE_FLAG_PROSE: Dict[str, str] = {
        "E1":  "Contradiction detected in current input. Acknowledge the tension directly.",
        "E6":  "Logic trap flagged. Do not follow the premise uncritically. Surface the trap gently.",
        "E14": "Socratic mode active (E14). Prioritise question-generation over direct answers.",
        "E7":  "Simulated opposition active (E7). Hold the strongest form of the opposing view.",
    }

    def _engine_flag_conditioning(self, stmm) -> str:
        lines = []
        for ex in stmm.brain_process_tracker.executions:
            if ex.skipped:
                continue
            prose = self._ENGINE_FLAG_PROSE.get(ex.engine_id)
            if prose and any(kw in ex.output_summary.lower() for kw in FLAG_KEYWORDS):
                lines.append(prose)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Component C — Memory contrast summary (v0.5)
    # ------------------------------------------------------------------

    def _memory_contrast_summary(self, stmm) -> str:
        """
        Brief memory contrast note when contradictions or unresolved
        queries are present.
        """
        mc = stmm.memory_contrast
        parts = []

        if mc.potential_contradictions:
            parts.append(
                f"{len(mc.potential_contradictions)} memory contradiction(s) detected. "
                "Handle with care — the user may have changed their mind."
            )
        if mc.unresolved_query_matches:
            parts.append(
                f"{len(mc.unresolved_query_matches)} unresolved query match(es) found. "
                "Consider whether this input resolves a prior open question."
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Component D — Operational context flag conditioning (v0.6)
    # ------------------------------------------------------------------

    #: Maps a context_flag key to conditioning prose injected into system message.
    _CONTEXT_FLAG_PROSE: Dict[str, str] = {
        "dream_mode":          "Dream mode active. Abstract associative thinking is valid. "
                               "Novel cross-domain connections are expected and welcome.",
        "autonomous_mode":     "Autonomous study mode. No live user is present. "
                               "Internal processing only — no user-facing response needed.",
        "e28_disabled":        "Emotion detection is disabled for this turn. "
                               "Do not infer emotional tone from user input.",
        "retroactive_contrast": "Retroactive contrast active. Compare current reasoning "
                                "against prior session patterns before responding.",
        "learning_reframe":    "Learning reframe active. Treat the input as teachable "
                               "material to be absorbed and integrated.",
        "confusion_override":  "Confusion signal active. Prioritise clarity and "
                               "step-by-step explanation over breadth.",
        "cb1_plasticity":      "CB1 plasticity flag active. Schema-breaking associations "
                               "are permitted; be open to unconventional framings.",
        "abstract_association": "Abstract association mode active. Look for structural "
                                "or analogical connections beyond literal content.",
    }

    def _context_flag_conditioning(self, bundle: Dict[str, Any]) -> str:
        """
        Return conditioning prose for active context_flags.

        Only flags listed in _CONTEXT_FLAG_PROSE produce output.
        Emphasis flags (emphasis:*) produce a single aggregated note.
        """
        ctx_flags = bundle.get("context_flags", {})
        if not ctx_flags:
            return ""

        lines = []
        emphasis_engines: List[str] = []

        for key in ctx_flags:
            if not isinstance(key, str):
                continue
            prose = self._CONTEXT_FLAG_PROSE.get(key)
            if prose:
                lines.append(prose)
            elif key.startswith("emphasis:"):
                emphasis_engines.append(key.split(":", 1)[1])
            elif key.startswith("dream_signal:"):
                sig = key.split(":", 1)[1]
                lines.append(
                    f"Dream signal active: {sig}. "
                    "Weight your response toward themes of "
                    + ("discovery and open-ended exploration."
                       if sig in ("curiosity", "wonder")
                       else "clarification and structural understanding.")
                )

        if emphasis_engines:
            lines.append(
                f"Processing emphasis on: {', '.join(emphasis_engines)}. "
                "Weight findings from these analytical dimensions more heavily."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Conversation history (v0.5 — same structure)
    # ------------------------------------------------------------------

    def _build_history(self, stmm) -> List[Dict[str, str]]:
        """
        Convert active_message_buffer.messages → OpenAI-format list.
        SpeakerID.USER → "user", SpeakerID.SYSTEM → "assistant".
        Ordered chronologically; latest user message is last.
        """
        from zados.memory.types import SpeakerID

        return [
            {
                "role": "user" if msg.speaker == SpeakerID.USER else "assistant",
                "content": msg.text,
            }
            for msg in stmm.active_message_buffer.messages
        ]

    # ------------------------------------------------------------------
    # ExtractorResult emotion formatting helper (v0.5)
    # ------------------------------------------------------------------

    def _format_extractor_emotions(self, extractor_result) -> str:
        """
        Format emotion saturations from ExtractorResult for inclusion
        in RG conditioning.
        """
        if extractor_result is None:
            return ""

        sats = getattr(extractor_result, "emotion_saturations", {})
        if not sats:
            return ""

        top3 = sorted(sats.items(), key=lambda x: x[1], reverse=True)[:3]
        parts = [f"{k}({v:.2f})" for k, v in top3 if v > 0.1]
        if not parts:
            return ""

        return "Extractor emotion saturations: " + ", ".join(parts)
