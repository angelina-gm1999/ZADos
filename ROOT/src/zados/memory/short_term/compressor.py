"""
Memory Exit Compressor (STMM §2.10).

Runs at the end of each processing cycle.  Packages STMM contents into a
MemoryPacket destined for MTMM, using the compression strategy from the spec:

  - Raw message text: lossless
  - Analysis results: compressed to key findings
  - Emotional state: emotion vector snapshot (lossless vector, lossy traces)
  - Neurochemical state: concentration snapshot
  - Reward evaluation: domain scores + flags (subscores dropped)
  - Engine trace: compressed to engine ID list + anomalies
  - Memory Contrast: matches above significance threshold only
"""
from __future__ import annotations

from zados.memory.short_term.store import STMMStore
from zados.memory.types import CompressionLevel, MemoryPacket, MemoryTier

# Threshold: contrast matches below this similarity score are dropped
_SIGNIFICANCE_THRESHOLD = 0.30


class MemoryExitCompressor:
    """Converts an STMMStore into a MemoryPacket ready for MTMM promotion."""

    def compress(self, stmm: STMMStore) -> MemoryPacket:
        packet = MemoryPacket(
            source_tier=MemoryTier.STMM,
            destination_tier=MemoryTier.MTMM,
        )

        # --- Lossless: raw text -------------------------------------------
        latest_user = stmm.active_message_buffer.latest_user()
        latest_sys  = stmm.active_message_buffer.latest_system()
        packet.user_message   = latest_user.text   if latest_user  else ""
        packet.system_response = latest_sys.text   if latest_sys   else ""
        packet.turn_index      = stmm.turn_index

        # --- Compressed: intention (primary only) -------------------------
        packet.intention = stmm.intention_analysis.primary_intention

        # --- Lossless vector: emotion state -------------------------------
        packet.emotion_vector = dict(stmm.emotion_detection.user_emotion_signals)
        packet.emotion_vector.update(
            {f"sys_{k}": v for k, v in stmm.emotion_detection.system_emotion_state.items()}
        )

        # --- Lossless concentrations: neurochemical snapshot --------------
        packet.neurochemical_snapshot = dict(stmm.cephalic_liquid_logger.nt_concentrations)

        # --- Lossy: domain-level reward scores (subscores dropped) --------
        packet.reward_scores = {
            domain: float(result.get("score", 0.0)) if isinstance(result, dict) else 0.0
            for domain, result in stmm.reward_evaluation.per_domain_results.items()
        }
        packet.flags = list(stmm.reward_evaluation.flags)

        # --- Standardized taxonomy tags (core/tags.py) --------------------
        try:
            from zados.core.tags import T

            # Pipeline origin
            active_mode = getattr(stmm.cortical_reflection, "active_mode", "")
            if active_mode:
                packet.flags.append(T.pipeline(
                    active_mode.lower().replace(" ", "_")
                ))

            # Primary intention
            intent_label = stmm.intention_analysis.primary_intention
            if intent_label:
                packet.flags.append(T.intent(intent_label.lower()))

            # Reward domain levels — logic, ethics, innovation, attunement
            per_domain = {}
            meta = stmm.reward_evaluation.meta_directive or {}
            meta_sub = meta.get("meta", {}) if isinstance(meta, dict) else {}
            if meta_sub:
                per_domain = meta_sub.get("per_domain_weighted_scores", {}) or {}
            for domain in ("logic", "ethics", "innovation", "attunement"):
                score = per_domain.get(domain, 0.0)
                if score > 0.0:
                    packet.flags.append(T.reward_from_score(domain, score))

            # Memory salience marker
            if packet.emotional_significance >= 0.60:
                packet.flags.append(T.mem("high_significance"))
            elif packet.emotional_significance < 0.15:
                packet.flags.append(T.mem("low_significance"))

        except Exception:
            pass  # tag stamping is non-critical; never block the pipeline

        # --- Compressed counts: detection results -------------------------
        packet.contradictions_detected = len(stmm.memory_contrast.potential_contradictions)
        packet.paradoxes_detected      = 0  # set by engine if available
        packet.unsolved_items_matched  = list(stmm.memory_contrast.unresolved_query_matches)

        # --- Significance-filtered contrast matches -----------------------
        # (stored in meta, not a dedicated field on MemoryPacket)

        # --- Emotional significance: max saturation level -----------------
        sat = stmm.emotion_detection.saturation_levels
        packet.emotional_significance = max(sat.values(), default=0.0)

        # --- Event flags from detection results ----------------------------
        try:
            from zados.core.tags import T
            if packet.contradictions_detected > 0:
                packet.flags.append(T.flag("contradiction"))
            if packet.unsolved_items_matched:
                packet.flags.append(T.flag("unsolved"))
        except Exception:
            pass

        # --- Compression metadata -----------------------------------------
        packet.compression_level = CompressionLevel.SEMANTIC

        return packet
