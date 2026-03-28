"""
ZA-DOS v0.6 — Dream Pipeline (spec §4.2 — REM analog).

Dream mode performs creative recombination of stagnated / unresolved items,
driven by the emotional state profile active during the dream phase.  It is
the natural continuation of REM processing: items that REM could not
consolidate (high confusion, stagnated questions) are handed off here for
abstract re-association.

Two interleaved functions:

  1. **Creative Recombination** — unsolved buffer items flagged as
     ``dream_candidate`` are processed with high CB1/GLU plasticity context
     flags that unlock abstract cross-domain connections.  A loose answer or
     novel angle, even partial, is enough to reduce the item's stagnation.

  2. **Retroactive Emotional Consolidation** — curiosity and confusion signals
     (the primary dream-phase drivers) bias the session's domain weights toward
     innovation and logic respectively, reinforcing the learning orientation
     most active during the session.

Dream-Phase Emotional Drivers (from MemoryPacket.neurochemical_snapshot)
-------------------------------------------------------------------------
  curiosity   — DA↑ ACh↑ CB1↑  → +innovation weight; drives novel associations
  confusion   — NE↑ GLU↑        → +logic weight; unresolved gaps need bridging
  wonder      — DA↑ CB1↑ 5ht↑  → +innovation weight; schema-breaking permitted
  perplexed   — DA↑ 5ht↑ gaba↓ → +logic +innovation; deep novelty signal

These are softer than REM's frustration/anxiety corrections — dream mode
leans into possibility rather than corrective re-weighting.

Triggered by ``/sleep dream`` (or automatically after REM if items remain).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from zados.core.tags import T
from zados.core.types import InputBundle, SessionState
from zados.memory.types import MemoryPacket

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight wrapper for identity-origin dream candidates
# ---------------------------------------------------------------------------

@dataclass
class _IdentityDreamCandidate:
    """Minimal candidate wrapper for identity questions pulled into dream."""
    question_text: str = ""
    question_id: str = ""
    tags: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_attempts: int = 0


# ---------------------------------------------------------------------------
# NT threshold tables for dream-phase signal detection
# ---------------------------------------------------------------------------

_DREAM_SIGNAL_THRESHOLDS: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {
    "curiosity": {
        "da":  (0.50, None),
        "ach": (0.40, None),
        "cb1": (0.30, None),
    },
    "confusion": {
        "ne":  (0.45, None),
        "glu": (0.35, None),
    },
    "wonder": {
        "da":  (0.55, None),
        "cb1": (0.35, None),
        "5ht": (0.35, None),
    },
    "perplexed": {
        "da":  (0.45, None),
        "5ht": (0.35, None),
        "gaba": (None, 0.35),   # disinhibition
    },
}

_DREAM_SIGNAL_DOMAIN_DELTAS: Dict[str, Dict[str, float]] = {
    "curiosity":  {"innovation": +0.07},
    "confusion":  {"logic": +0.06},
    "wonder":     {"innovation": +0.09},
    "perplexed":  {"logic": +0.04, "innovation": +0.05},
}

# Maximum dream candidates to attempt recombination on per run.
_MAX_DREAM_CANDIDATES = 6


# ---------------------------------------------------------------------------
# Dream result summary
# ---------------------------------------------------------------------------

@dataclass
class DreamResult:
    """Summary of one Dream pipeline run."""
    session_id: str = ""
    candidates_found: int = 0
    candidates_processed: int = 0
    novel_connections: int = 0
    dominant_signals: List[str] = field(default_factory=list)
    domain_weight_adjustments: Dict[str, float] = field(default_factory=dict)
    processing_time_s: float = 0.0


# ===========================================================================
# DreamPipeline
# ===========================================================================

class DreamPipeline:
    """Dream mode — creative recombination + retroactive emotional consolidation.

    Parameters
    ----------
    answer_pipeline : AnswerPipeline, optional
        Used to process dream candidates through abstract re-association.
    memory : MemoryLayer, optional
        MTMM/LTMM access for reading packets and writing novel connections.
    unsolved_buffer : UnsolvedBuffer, optional
        Source of dream-candidate items.
    neurochem_engine : NeurochemicalEngine, optional
        Read-only; used to read live dream-phase NT state if packets are sparse.
    """

    def __init__(
        self,
        answer_pipeline: Any = None,
        memory: Any = None,
        unsolved_buffer: Any = None,
        neurochem_engine: Any = None,
        journal_store: Any = None,
    ) -> None:
        self._pipeline = answer_pipeline
        self._memory = memory
        self._unsolved = unsolved_buffer
        self._neurochem = neurochem_engine
        self._journal_store = journal_store

    # ===================================================================
    # Main entry point
    # ===================================================================

    def process(self, session: SessionState) -> Dict[str, Any]:
        """Run dream processing: creative recombination + weight consolidation.

        Parameters
        ----------
        session : SessionState

        Returns
        -------
        Dict[str, Any]
            Summary of dream run results.
        """
        log.info("Dream Pipeline: session %s — creative recombination.", session.session_id)
        start_time = time.time()

        result = DreamResult(session_id=session.session_id)

        # --- Phase 0: Gather dream candidates + build emotional driver profile ---
        candidates = self._phase0_gather_candidates()
        result.candidates_found = len(candidates)

        signal_profile = self._phase1_build_signal_profile()
        result.dominant_signals = [
            sig for sig, w in sorted(
                signal_profile.items(), key=lambda x: x[1], reverse=True
            )
            if w > 0.1
        ][:3]

        # --- Phase 2: Retroactive domain weight adjustments ---
        adjustments = self._phase2_compute_adjustments(signal_profile)
        result.domain_weight_adjustments = adjustments
        if adjustments and hasattr(session, "learned_domain_weights"):
            self._apply_domain_adjustments(session, adjustments)
            log.info("Dream: domain weight adjustments applied: %s",
                     {k: round(v, 3) for k, v in adjustments.items()})

        # --- Phase 3: Creative recombination of dream candidates ---
        if candidates:
            novel = self._phase3_recombine(candidates, signal_profile, session)
            result.candidates_processed = len(candidates[:_MAX_DREAM_CANDIDATES])
            result.novel_connections = novel
        else:
            log.info("Dream Pipeline: no dream candidates — skipping recombination.")

        log.info("Dream Pipeline complete: %d candidates, %d novel connections, signals=%s",
                 result.candidates_found, result.novel_connections, result.dominant_signals)

        # --- Journal write ---
        self._write_journal(result, session)

        return self._build_return(result, start_time)

    # ===================================================================
    # Phase 0: Gather dream candidates
    # ===================================================================

    def _phase0_gather_candidates(self) -> List[Any]:
        """Collect dream candidates, sorted by origin priority.

        Academic-origin items are deprioritised (sorted last) so the
        dream pipeline favours identity and general items for creative
        recombination.  Academic items are better handled by REM's
        logic-boosted consolidation.

        Also pulls identity-relevant questions from GeneralQuestionStore
        (if available) and tags them ``origin:identity``.
        """
        candidates: List[Any] = []

        # Source 1: UnsolvedBuffer dream candidates
        if self._unsolved is not None:
            try:
                active = self._unsolved.get_active()
                candidates.extend(
                    q for q in active
                    if "dream_candidate" in getattr(q, "tags", [])
                )
            except Exception:
                log.debug("Dream: unsolved buffer read failed.", exc_info=True)

        # Source 2: Identity-relevant questions from GeneralQuestionStore
        if self._memory is not None:
            gq_store = getattr(
                getattr(self._memory, "thoughts", None), "general_questions", None
            )
            if gq_store is not None:
                try:
                    from zados.core.tags import T
                    for q in gq_store.get_unresolved():
                        scope = getattr(q, "scope_tag", "")
                        if scope == "identity":
                            # Wrap as a lightweight candidate-like object
                            candidates.append(_IdentityDreamCandidate(
                                question_text=getattr(q, "question_text", str(q)),
                                question_id=getattr(q, "question_id", ""),
                                tags=[T.origin("identity"), "dream_candidate"],
                            ))
                except Exception:
                    log.debug("Dream: identity question scan failed.", exc_info=True)

        # Sort: identity and general first, academic last
        def _origin_priority(item: Any) -> int:
            tags = getattr(item, "tags", [])
            if "origin:academic" in tags:
                return 2   # lowest priority for dream
            if "origin:identity" in tags:
                return 0   # highest priority for dream
            return 1       # general / dialectic — middle

        candidates.sort(key=_origin_priority)
        return candidates

    # ===================================================================
    # Phase 1: Build emotional driver profile
    # ===================================================================

    def _phase1_build_signal_profile(self) -> Dict[str, float]:
        """Build dream-phase emotional signal profile.

        Combines two sources:
        1. Recent MTMM packet NT snapshots (retroactive from session)
        2. Live neurochem readout if available (current dream-phase state)
        """
        signal_profile: Dict[str, float] = {}

        # Source 1: MTMM packets
        packets = self._read_recent_packets(limit=20)
        if packets:
            signal_profile = self._aggregate_dream_signals(packets)

        # Source 2: Live neurochem readout (supplements packet data)
        live_signals = self._read_live_dream_signals()
        for sig, strength in live_signals.items():
            # Average with packet-based estimate (or use directly if no packets)
            existing = signal_profile.get(sig, 0.0)
            signal_profile[sig] = (existing + strength) / 2.0 if existing else strength

        return signal_profile

    def _read_recent_packets(self, limit: int = 20) -> List[MemoryPacket]:
        """Read the most recent MTMM packets."""
        if self._memory is None:
            return []
        try:
            logger = getattr(getattr(self._memory, "mtmm", None), "logger", None)
            if logger is None:
                return []
            all_pkts = logger.get_all()
            return all_pkts[-limit:] if len(all_pkts) > limit else all_pkts
        except Exception:
            log.debug("Dream: MTMM read failed.", exc_info=True)
            return []

    def _aggregate_dream_signals(self, packets: List[MemoryPacket]) -> Dict[str, float]:
        """Detect dream-relevant signals across packets."""
        signal_sums: Dict[str, float] = {}
        total_weight = 0.0

        for pkt in packets:
            weight = max(0.1, pkt.emotional_significance)
            total_weight += weight
            nt = pkt.neurochemical_snapshot

            if nt:
                detected = self._detect_dream_signals_from_nt(nt)
            else:
                detected = self._detect_dream_signals_from_emotions(pkt.emotion_vector)

            for sig, strength in detected.items():
                signal_sums[sig] = signal_sums.get(sig, 0.0) + strength * weight

        if total_weight == 0.0:
            return {}

        return {sig: v / total_weight for sig, v in signal_sums.items()}

    @staticmethod
    def _detect_dream_signals_from_nt(nt: Dict[str, float]) -> Dict[str, float]:
        detected: Dict[str, float] = {}
        for signal, conditions in _DREAM_SIGNAL_THRESHOLDS.items():
            matches = 0
            total_strength = 0.0
            for nt_key, (lo, hi) in conditions.items():
                val = nt.get(nt_key, nt.get(nt_key.upper(), 0.0))
                nt_match = True
                if lo is not None and val < lo:
                    nt_match = False
                if hi is not None and val > hi:
                    nt_match = False
                if nt_match:
                    if lo is not None:
                        strength = min(1.0, (val - lo) / max(lo, 0.01))
                    elif hi is not None:
                        strength = min(1.0, (hi - val) / max(hi, 0.01))
                    else:
                        strength = val
                    matches += 1
                    total_strength += strength
            n = len(conditions)
            if matches >= max(1, n - 1):
                detected[signal] = total_strength / n
        return detected

    @staticmethod
    def _detect_dream_signals_from_emotions(emotion_vector: Dict[str, float]) -> Dict[str, float]:
        detected: Dict[str, float] = {}
        label_map = {
            "curious":    "curiosity",
            "confused":   "confusion",
            "wonder":     "wonder",
            "perplexed":  "perplexed",
        }
        for label, signal in label_map.items():
            val = emotion_vector.get(label, 0.0)
            if val > 0.25:
                detected[signal] = val
        return detected

    def _read_live_dream_signals(self) -> Dict[str, float]:
        """Read live NT state from neurochem engine to supplement signal profile."""
        if self._neurochem is None:
            return {}
        try:
            readout = self._neurochem.get_neurosymbolic_readout()
            metrics = readout if isinstance(readout, dict) else (
                readout.as_dict() if hasattr(readout, "as_dict") else {}
            )
            signals: Dict[str, float] = {}
            # Curiosity proxy: high novelty_seeking / low cognitive_rigidity
            novelty = metrics.get("novelty_seeking", 0.0)
            rigidity = metrics.get("cognitive_rigidity", 0.5)
            if novelty > 0.5 or rigidity < 0.3:
                signals["curiosity"] = max(novelty, 0.5 - rigidity)
            # Confusion proxy: high uncertainty / low precision
            uncertainty = metrics.get("uncertainty", 0.0)
            precision = metrics.get("precision", 0.5)
            if uncertainty > 0.5 or precision < 0.35:
                signals["confusion"] = max(uncertainty, 0.5 - precision)
            return signals
        except Exception:
            log.debug("Dream: neurochem readout failed.", exc_info=True)
            return {}

    # ===================================================================
    # Phase 2: Compute retroactive domain weight adjustments
    # ===================================================================

    @staticmethod
    def _phase2_compute_adjustments(
        signal_profile: Dict[str, float],
    ) -> Dict[str, float]:
        adjustments: Dict[str, float] = {}
        for signal, strength in signal_profile.items():
            for domain, base_delta in _DREAM_SIGNAL_DOMAIN_DELTAS.get(signal, {}).items():
                adjustments[domain] = adjustments.get(domain, 0.0) + base_delta * strength
        return adjustments

    @staticmethod
    def _apply_domain_adjustments(
        session: SessionState,
        adjustments: Dict[str, float],
    ) -> None:
        current = session.learned_domain_weights
        for domain, delta in adjustments.items():
            old_val = current.get(domain, 0.5)
            current[domain] = max(0.0, min(1.0, old_val + delta))

    # ===================================================================
    # Phase 3: Creative recombination of dream candidates
    # ===================================================================

    def _phase3_recombine(
        self,
        candidates: List[Any],
        signal_profile: Dict[str, float],
        session: SessionState,
    ) -> int:
        """Attempt abstract re-association of dream candidates.

        Each candidate is processed through the answer pipeline with dream-mode
        context flags (CB1 plasticity, abstract association enabled).  Any
        result longer than a threshold is counted as a novel connection and
        written to LTMM.

        Returns number of novel connections found.
        """
        if self._pipeline is None:
            log.info("Dream: no answer pipeline available; skipping recombination.")
            return 0

        novel = 0
        top_candidates = candidates[:_MAX_DREAM_CANDIDATES]

        # Build dream context flags from signal profile
        context_flags: Dict[str, Any] = {
            "dream_mode": True,
            "cb1_plasticity": True,     # schema flexibility
            "abstract_association": True,
        }
        for sig in signal_profile:
            context_flags[f"dream_signal:{sig}"] = True

        for q in top_candidates:
            question_text = getattr(q, "question_text", str(q))
            try:
                bundle = InputBundle(
                    raw_text=f"[DREAM RECOMBINATION] {question_text}",
                    active_mode="dream",
                )
                bundle.context_flags.update(context_flags)
                # Add origin-specific context flags for neurochem modulation
                q_tags = getattr(q, "tags", [])
                if "origin:identity" in q_tags:
                    bundle.context_flags["identity_salience"] = True
                    bundle.context_flags["oxt_boost"] = True
                elif "origin:academic" in q_tags:
                    bundle.context_flags["academic_salience"] = True

                dream_session = SessionState(
                    session_id=f"dream_{session.session_id}",
                    session_mode="dream",
                    initial_mode="dream",
                )

                result = self._pipeline.process_turn(bundle, dream_session)
                answer = getattr(result, "final_answer", "") or ""

                if len(answer) > 40:
                    novel += 1
                    q_id = getattr(q, "question_id", None)
                    if q_id and self._unsolved is not None:
                        try:
                            self._unsolved.mark_attempted(
                                q_id,
                                partial_answer=f"[DREAM] {answer[:200]}",
                            )
                        except Exception:
                            log.debug("Dream: mark_attempted failed for %s.", q_id)
                    # Propagate origin tags to the LTMM connection
                    origin_flags = [
                        t for t in q_tags if t.startswith("origin:")
                    ]
                    self._write_dream_connection(
                        question_text, answer, session,
                        extra_flags=origin_flags,
                    )

            except Exception:
                log.debug("Dream: recombination failed for candidate.", exc_info=True)

        log.info("Dream Phase 3: %d/%d candidates produced novel connections.",
                 novel, len(top_candidates))
        return novel

    def _write_dream_connection(
        self,
        question: str,
        answer: str,
        session: SessionState,
        extra_flags: Optional[List[str]] = None,
    ) -> None:
        """Write a novel dream connection as a MemoryPacket to LTMM."""
        if self._memory is None:
            return
        try:
            from zados.memory.long_term.store import LTMMEntry, Granularity
            from zados.memory.types import MemoryPacket, MemoryTier
            from datetime import datetime

            flags = ["dream", "novel_association"]
            if extra_flags:
                flags.extend(extra_flags)

            is_identity = any(f == "origin:identity" for f in (extra_flags or []))

            pkt = MemoryPacket(
                source_tier=MemoryTier.STMM,
                destination_tier=MemoryTier.LTMM,
                user_message=question,
                system_response=answer,
                intention="dream_connection",
                flags=flags,
                emotional_significance=0.7 if is_identity else 0.6,
            )
            entry = LTMMEntry(
                packet=pkt,
                granularity=Granularity.SEMANTIC,
                relevance_score=0.8 if is_identity else 0.7,
                identity_relevant=is_identity,
            )
            ltmm = getattr(self._memory, "ltmm", None)
            if ltmm is not None:
                ltmm.write(entry)
        except Exception:
            log.debug("Dream: LTMM write for novel connection failed.", exc_info=True)

    # ===================================================================
    # Journal write
    # ===================================================================

    def _write_journal(self, result: DreamResult, session: SessionState) -> None:
        """Write a REM_COMPLETE journal entry for the dream run (if store available)."""
        if self._journal_store is None:
            return
        try:
            from zados.memory.long_term.journal.entry import JournalEntry, JournalTrigger
            from zados.core.tags import T

            notes = (
                [f"pipeline:dream",
                 f"candidates_found:{result.candidates_found}",
                 f"candidates_processed:{result.candidates_processed}",
                 f"novel_connections:{result.novel_connections}"]
                + [f"signal:{s}" for s in result.dominant_signals]
            )

            prose = (
                f"Dream recombination complete. "
                f"{result.candidates_processed} dream candidates processed, "
                f"{result.novel_connections} novel connections found. "
            )
            if result.dominant_signals:
                prose += (
                    f"Active dream-phase signals: "
                    f"{', '.join(result.dominant_signals)}. "
                )
            if result.domain_weight_adjustments:
                prose += (
                    "Domain orientation nudged: "
                    + ", ".join(
                        f"{d} {'+' if v >= 0 else ''}{v:.3f}"
                        for d, v in result.domain_weight_adjustments.items()
                    ) + "."
                )

            entry = JournalEntry(
                session_id=session.session_id,
                trigger=JournalTrigger.REM_COMPLETE,
                trigger_source="dream_pipeline",
                prose=prose,
                pipeline_notes=notes,
                tags=T.pipeline_tags_for_sleep("dream", result.dominant_signals),
            )
            self._journal_store.write(entry)
            log.info("Dream: journal entry written (trigger=REM_COMPLETE).")
        except Exception:
            log.debug("Dream: journal write failed.", exc_info=True)

    # ===================================================================
    # Return builder
    # ===================================================================

    @staticmethod
    def _build_return(result: DreamResult, start_time: float) -> Dict[str, Any]:
        return {
            "status": "completed",
            "session_id": result.session_id,
            "processing_time_s": round(time.time() - start_time, 2),
            "candidates_found": result.candidates_found,
            "candidates_processed": result.candidates_processed,
            "novel_connections": result.novel_connections,
            "dominant_signals": result.dominant_signals,
            "domain_weight_adjustments": {
                k: round(v, 4) for k, v in result.domain_weight_adjustments.items()
            },
        }
