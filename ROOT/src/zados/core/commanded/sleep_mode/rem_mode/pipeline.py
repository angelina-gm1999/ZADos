"""
ZA-DOS v0.6 — REM Pipeline (spec §4.1).

Performs two interleaved functions during sleep REM processing:

  1. **Memory Consolidation** — MTMM → LTMM promotion of high-value packets.
     Emotionally significant and reward-salient packets from the current
     session are written to LTMM; low-importance packets are left to decay.

  2. **Retroactive Learning** — Domain weight self-adjustment based on
     learning-relevant emotional signals accumulated across the session.
     Mirrors the Homework pipeline's NT-based deficit profiling but driven
     by the session's own memory packets rather than a learning log.

Retroactive Learning Emotional Signals (from MemoryPacket.neurochemical_snapshot)
----------------------------------------------------------------------------------
  frustration  — NE↑ DA↑ COR↑  → raise logic + ethics weights (what went wrong?)
  curiosity    — DA↑ ACh↑ CB1↑ → raise innovation weight (pursue discoveries)
  confusion    — NE↑ GLU↑       → raise logic weight (need stronger reasoning)
  boredom      — DA↓ NE↓        → dampen all weights slightly (low engagement)
  anxiety      — NE↑ COR↑       → raise ethics weight (risk-aware turn)
  overwhelmed  — NE↑↑ COR↑↑    → soft-dampen all weights (system overload)

Domain weight adjustments are accumulated into ``session.learned_domain_weights``
(same target as E17 in Phase 7), clamped to [0.0, 1.0].

Triggered by ``/sleep rem`` command.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from zados.core.types import SessionState
from zados.memory.types import MemoryPacket

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NT threshold tables for learning-signal detection
# ---------------------------------------------------------------------------

# Each signal: {nt_key: (min_threshold, max_threshold)}
# None = no bound on that side.
# Keys match MemoryPacket.neurochemical_snapshot convention (lowercase).
_LEARNING_SIGNAL_THRESHOLDS: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {
    "frustration": {
        "ne":  (0.50, None),
        "da":  (0.40, None),
        "cor": (0.40, None),
    },
    "curiosity": {
        "da":  (0.50, None),
        "ach": (0.40, None),
        "cb1": (0.30, None),
    },
    "confusion": {
        "ne":  (0.45, None),
        "glu": (0.35, None),
    },
    "boredom": {
        "da":  (None, 0.30),  # suppressed
        "ne":  (None, 0.30),  # suppressed
    },
    "anxiety": {
        "ne":  (0.55, None),
        "cor": (0.50, None),
    },
    "overwhelmed": {
        "ne":  (0.65, None),
        "cor": (0.60, None),
    },
}

# Domain weight deltas per detected signal (added to learned_domain_weights).
# Positive = raise, negative = lower.  Applied proportionally to signal strength.
_SIGNAL_DOMAIN_DELTAS: Dict[str, Dict[str, float]] = {
    "frustration": {"logic": +0.06,  "ethics": +0.04},
    "curiosity":   {"innovation": +0.08},
    "confusion":   {"logic": +0.07},
    "boredom":     {"logic": -0.03, "ethics": -0.03, "innovation": -0.03, "attunement": -0.03},
    "anxiety":     {"ethics": +0.05},
    "overwhelmed": {"logic": -0.02, "ethics": -0.02, "innovation": -0.02, "attunement": -0.02},
}

# Minimum emotional_significance to promote a packet to LTMM.
_LTMM_SIGNIFICANCE_THRESHOLD = 0.45

# Minimum reward score average to promote a packet (secondary gate).
_LTMM_REWARD_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# REM result summary
# ---------------------------------------------------------------------------

@dataclass
class REMResult:
    """Summary of one REM pipeline run."""
    session_id: str = ""
    packets_scanned: int = 0
    packets_consolidated: int = 0
    dominant_signals: List[str] = field(default_factory=list)
    domain_weight_adjustments: Dict[str, float] = field(default_factory=dict)
    processing_time_s: float = 0.0


# ===========================================================================
# REMPipeline
# ===========================================================================

class REMPipeline:
    """REM sleep mode — memory consolidation + retroactive learning.

    Parameters
    ----------
    answer_pipeline : AnswerPipeline, optional
        Reserved for future replay-based processing; not used in current impl.
    memory : MemoryLayer, optional
        MTMM/LTMM access for consolidation.
    neurochem_engine : NeurochemicalEngine, optional
        Read-only access to current NT state (supplements packet analysis).
    """

    def __init__(
        self,
        answer_pipeline: Any = None,
        memory: Any = None,
        neurochem_engine: Any = None,
        journal_store: Any = None,
    ) -> None:
        self._pipeline = answer_pipeline
        self._memory = memory
        self._neurochem = neurochem_engine
        self._journal_store = journal_store

    # ===================================================================
    # Main entry point
    # ===================================================================

    def process(self, session: SessionState) -> Dict[str, Any]:
        """Run REM consolidation + retroactive learning.

        Parameters
        ----------
        session : SessionState

        Returns
        -------
        Dict[str, Any]
            Summary of consolidation + learning adjustments.
        """
        log.info("REM Pipeline: session %s — starting consolidation + retroactive learning.",
                 session.session_id)
        start_time = time.time()

        rem_result = REMResult(session_id=session.session_id)

        # --- Phase 0: Read MTMM packets ---
        packets = self._phase0_read_packets()
        rem_result.packets_scanned = len(packets)

        if not packets:
            log.info("REM Pipeline: no MTMM packets found, skipping.")
            return self._build_return(rem_result, start_time)

        # --- Phase 1: Score packets for emotional learning signals ---
        scored_packets = self._phase1_score_emotional_signals(packets)

        # --- Phase 2: Aggregate session emotional signal profile ---
        signal_profile = self._phase2_aggregate_signal_profile(scored_packets)
        rem_result.dominant_signals = [
            sig for sig, weight in sorted(
                signal_profile.items(), key=lambda x: x[1], reverse=True
            )
            if weight > 0.1
        ][:4]

        # --- Phase 2.5: Origin-tagged concept weight boosts ---
        origin_adjustments = self._origin_based_adjustments()
        if origin_adjustments:
            for domain, delta in origin_adjustments.items():
                signal_profile.setdefault("_origin_boost", 0.0)
            log.info("REM: origin-based adjustments detected: %s", origin_adjustments)

        # --- Phase 3: Compute + apply retroactive domain weight adjustments ---
        adjustments = self._phase3_compute_adjustments(signal_profile)
        # Merge origin-based adjustments into the total
        for domain, delta in origin_adjustments.items():
            adjustments[domain] = adjustments.get(domain, 0.0) + delta
        rem_result.domain_weight_adjustments = adjustments
        if adjustments and hasattr(session, "learned_domain_weights"):
            self._apply_domain_adjustments(session, adjustments)
            log.info("REM: domain weight adjustments applied: %s",
                     {k: round(v, 3) for k, v in adjustments.items()})

        # --- Phase 4: MTMM → LTMM consolidation ---
        consolidated = self._phase4_consolidate(packets)
        rem_result.packets_consolidated = consolidated

        log.info("REM Pipeline complete: %d/%d packets consolidated, signals=%s",
                 consolidated, len(packets), rem_result.dominant_signals)

        # --- Journal write ---
        self._write_journal(rem_result, session)

        return self._build_return(rem_result, start_time)

    # ===================================================================
    # Phase 0: Read MTMM packets
    # ===================================================================

    def _phase0_read_packets(self) -> List[MemoryPacket]:
        """Return all MTMM packets logged this session."""
        if self._memory is None:
            return []
        try:
            logger = getattr(getattr(self._memory, "mtmm", None), "logger", None)
            if logger is None:
                return []
            return logger.get_all()
        except Exception:
            log.debug("REM: MTMM read failed.", exc_info=True)
            return []

    # ===================================================================
    # Phase 1: Score each packet for learning-relevant emotional signals
    # ===================================================================

    def _phase1_score_emotional_signals(
        self,
        packets: List[MemoryPacket],
    ) -> List[Tuple[MemoryPacket, Dict[str, float]]]:
        """Detect learning-relevant emotional signals in each packet.

        Returns
        -------
        List of (packet, signal_scores) where signal_scores maps
        signal_name → detected strength [0, 1].
        """
        scored: List[Tuple[MemoryPacket, Dict[str, float]]] = []

        for pkt in packets:
            nt = pkt.neurochemical_snapshot
            if not nt:
                # Packet was recompressed — use emotion_vector fallback
                signal_scores = self._detect_signals_from_emotions(pkt.emotion_vector)
            else:
                signal_scores = self._detect_signals_from_nt(nt)
            scored.append((pkt, signal_scores))

        return scored

    @staticmethod
    def _detect_signals_from_nt(nt: Dict[str, float]) -> Dict[str, float]:
        """Detect learning signals from NT concentration snapshot.

        Returns signal_name → strength in [0, 1].
        """
        detected: Dict[str, float] = {}

        for signal, conditions in _LEARNING_SIGNAL_THRESHOLDS.items():
            matches = 0
            total_strength = 0.0

            for nt_key, (lo, hi) in conditions.items():
                val = nt.get(nt_key, nt.get(nt_key.upper(), 0.0))
                nt_match = True
                strength = 0.0

                if lo is not None and val < lo:
                    nt_match = False
                if hi is not None and val > hi:
                    nt_match = False

                if nt_match:
                    # Strength = how far past the threshold
                    if lo is not None:
                        strength = min(1.0, (val - lo) / max(lo, 0.01))
                    elif hi is not None:
                        strength = min(1.0, (hi - val) / max(hi, 0.01))
                    else:
                        strength = val
                    matches += 1
                    total_strength += strength

            n_conditions = len(conditions)
            if matches >= max(1, n_conditions - 1):   # allow one miss
                detected[signal] = total_strength / n_conditions

        return detected

    @staticmethod
    def _detect_signals_from_emotions(
        emotion_vector: Dict[str, float],
    ) -> Dict[str, float]:
        """Fallback: detect learning signals from emotion_vector labels."""
        detected: Dict[str, float] = {}
        # Direct label matches for recompressed packets
        label_map = {
            "frustrated": "frustration",
            "frustrated_learning": "frustration",
            "curious": "curiosity",
            "confused": "confusion",
            "bored": "boredom",
            "anxious": "anxiety",
            "overwhelmed": "overwhelmed",
        }
        for label, signal in label_map.items():
            val = emotion_vector.get(label, 0.0)
            if val > 0.25:
                detected[signal] = val
        return detected

    # ===================================================================
    # Phase 2: Aggregate signal profile across all packets
    # ===================================================================

    @staticmethod
    def _phase2_aggregate_signal_profile(
        scored_packets: List[Tuple[MemoryPacket, Dict[str, float]]],
    ) -> Dict[str, float]:
        """Average signal strengths across all packets, weighted by emotional significance."""
        if not scored_packets:
            return {}

        signal_sums: Dict[str, float] = {}
        signal_counts: Dict[str, int] = {}
        total_weight = 0.0

        for pkt, signals in scored_packets:
            weight = max(0.1, pkt.emotional_significance)  # floor at 0.1
            total_weight += weight
            for sig, strength in signals.items():
                signal_sums[sig] = signal_sums.get(sig, 0.0) + strength * weight
                signal_counts[sig] = signal_counts.get(sig, 0) + 1

        if total_weight == 0.0:
            return {}

        return {
            sig: signal_sums[sig] / total_weight
            for sig in signal_sums
        }

    # ===================================================================
    # Phase 3: Compute retroactive domain weight adjustments
    # ===================================================================

    @staticmethod
    def _phase3_compute_adjustments(
        signal_profile: Dict[str, float],
    ) -> Dict[str, float]:
        """Map session emotional signal profile to domain weight adjustments.

        The adjustment magnitude is proportional to signal strength.
        """
        adjustments: Dict[str, float] = {}

        for signal, strength in signal_profile.items():
            deltas = _SIGNAL_DOMAIN_DELTAS.get(signal, {})
            for domain, base_delta in deltas.items():
                adjustments[domain] = (
                    adjustments.get(domain, 0.0) + base_delta * strength
                )

        return adjustments

    @staticmethod
    def _apply_domain_adjustments(
        session: SessionState,
        adjustments: Dict[str, float],
    ) -> None:
        """Write adjustments to session.learned_domain_weights (clamped [0,1])."""
        current = session.learned_domain_weights
        for domain, delta in adjustments.items():
            old_val = current.get(domain, 0.5)
            current[domain] = max(0.0, min(1.0, old_val + delta))

    # ===================================================================
    # Origin-based adjustments (academic / identity concept boosts)
    # ===================================================================

    # Domain weight boosts per origin tag.  Applied additively.
    _ORIGIN_DOMAIN_BOOSTS: Dict[str, Dict[str, float]] = {
        "academic":  {"logic": +0.06, "ethics": +0.02},
        "identity":  {"ethics": +0.06, "attunement": +0.05},
        "dialectic": {"logic": +0.04, "ethics": +0.03},
    }

    def _origin_based_adjustments(self) -> Dict[str, float]:
        """Scan AcademicBufferStore and identity questions for origin tags.

        Returns aggregated domain weight boosts proportional to the
        number of unresolved origin-tagged items found.
        """
        adjustments: Dict[str, float] = {}
        if self._memory is None:
            return adjustments

        origin_counts: Dict[str, int] = {}

        # Source 1: AcademicBufferStore dream candidates
        academic_buf = getattr(
            getattr(self._memory, "knowledge", None), "academic_buffer", None
        )
        if academic_buf is not None:
            for entry in academic_buf.get_dream_candidates():
                origin_counts["academic"] = origin_counts.get("academic", 0) + 1

        # Source 2: Identity-relevant questions (general_questions with identity scope)
        gq_store = getattr(
            getattr(self._memory, "thoughts", None), "general_questions", None
        )
        if gq_store is not None:
            try:
                for q in gq_store.get_unresolved():
                    scope = getattr(q, "scope_tag", "")
                    if scope == "identity":
                        origin_counts["identity"] = origin_counts.get("identity", 0) + 1
            except Exception:
                log.debug("REM: identity question scan failed.", exc_info=True)

        # Source 3: Identity conclusions store (unresolved / pending entries)
        conclusions_store = getattr(
            getattr(self._memory, "identity", None), "conclusions", None
        )
        if conclusions_store is not None:
            try:
                all_conclusions = conclusions_store.get_all()
                for c in all_conclusions:
                    status = getattr(c, "status", "")
                    if status in ("pending", "challenged"):
                        origin_counts["identity"] = (
                            origin_counts.get("identity", 0) + 1
                        )
            except Exception:
                log.debug("REM: identity conclusion scan failed.", exc_info=True)

        # Compute boosts: diminishing returns per item (sqrt scaling)
        import math
        for origin, count in origin_counts.items():
            boosts = self._ORIGIN_DOMAIN_BOOSTS.get(origin, {})
            scale = min(1.0, math.sqrt(count) / 3.0)  # 1 item → 0.33x, 9+ → 1.0x
            for domain, base_delta in boosts.items():
                adjustments[domain] = (
                    adjustments.get(domain, 0.0) + base_delta * scale
                )

        if origin_counts:
            log.info("REM: origin-tagged items found: %s", origin_counts)

        return adjustments

    # ===================================================================
    # Phase 4: MTMM → LTMM consolidation
    # ===================================================================

    def _phase4_consolidate(self, packets: List[MemoryPacket]) -> int:
        """Promote emotionally or reward-significant packets to LTMM.

        Returns number of packets actually written.
        """
        ltmm = getattr(self._memory, "ltmm", None) if self._memory else None
        if ltmm is None:
            return 0

        consolidated = 0
        for pkt in packets:
            if self._should_consolidate(pkt):
                try:
                    from zados.memory.long_term.store import LTMMEntry, Granularity
                    entry = LTMMEntry(
                        packet=pkt,
                        granularity=Granularity.SEMANTIC,
                        relevance_score=min(1.0, pkt.emotional_significance + 0.2),
                        identity_relevant="identity" in pkt.flags,
                    )
                    ltmm.write(entry)
                    consolidated += 1
                except Exception:
                    log.debug("REM: LTMM write failed for packet %s.", pkt.packet_id, exc_info=True)

        log.info("REM Phase 4: %d/%d packets promoted to LTMM.", consolidated, len(packets))
        return consolidated

    @staticmethod
    def _should_consolidate(pkt: MemoryPacket) -> bool:
        """Return True if this packet is worth promoting to LTMM."""
        # Primary gate: emotional significance
        if pkt.emotional_significance >= _LTMM_SIGNIFICANCE_THRESHOLD:
            return True
        # Secondary gate: reward scores (any domain scored well)
        if pkt.reward_scores:
            avg_reward = sum(pkt.reward_scores.values()) / len(pkt.reward_scores)
            if avg_reward >= _LTMM_REWARD_THRESHOLD:
                return True
        # Tertiary gate: contradiction/paradox detected (high information value)
        if pkt.contradictions_detected > 0 or pkt.paradoxes_detected > 0:
            return True
        return False

    # ===================================================================
    # Journal write
    # ===================================================================

    def _write_journal(self, result: REMResult, session: SessionState) -> None:
        """Write a REM_COMPLETE journal entry to JournalStore (if available)."""
        if self._journal_store is None:
            return
        try:
            from zados.memory.long_term.journal.entry import JournalEntry, JournalTrigger
            from zados.core.tags import T

            adj_parts = [
                f"{domain}:{delta:+.3f}"
                for domain, delta in result.domain_weight_adjustments.items()
            ]
            notes = (
                [f"pipeline:rem",
                 f"packets_scanned:{result.packets_scanned}",
                 f"packets_consolidated:{result.packets_consolidated}"]
                + [f"signal:{s}" for s in result.dominant_signals]
                + ([f"adjustments:{' '.join(adj_parts)}"] if adj_parts else [])
            )

            prose = (
                f"REM consolidation complete. "
                f"{result.packets_consolidated} of {result.packets_scanned} packets "
                f"promoted to long-term memory. "
            )
            if result.dominant_signals:
                prose += (
                    f"Dominant learning signals this session: "
                    f"{', '.join(result.dominant_signals)}. "
                )
            if result.domain_weight_adjustments:
                prose += (
                    f"Domain weight adjustments applied: "
                    + ", ".join(
                        f"{d} {'+' if v >= 0 else ''}{v:.3f}"
                        for d, v in result.domain_weight_adjustments.items()
                    ) + "."
                )

            entry = JournalEntry(
                session_id=session.session_id,
                trigger=JournalTrigger.REM_COMPLETE,
                trigger_source="rem_pipeline",
                prose=prose,
                pipeline_notes=notes,
                tags=T.pipeline_tags_for_sleep("rem", result.dominant_signals),
            )
            self._journal_store.write(entry)
            log.info("REM: journal entry written (trigger=REM_COMPLETE).")
        except Exception:
            log.debug("REM: journal write failed.", exc_info=True)

    # ===================================================================
    # Return builder
    # ===================================================================

    @staticmethod
    def _build_return(result: REMResult, start_time: float) -> Dict[str, Any]:
        return {
            "status": "completed",
            "session_id": result.session_id,
            "processing_time_s": round(time.time() - start_time, 2),
            "packets_scanned": result.packets_scanned,
            "packets_consolidated": result.packets_consolidated,
            "dominant_signals": result.dominant_signals,
            "domain_weight_adjustments": {
                k: round(v, 4) for k, v in result.domain_weight_adjustments.items()
            },
        }
