"""
ZA-DOS v0.6 — Reflective Pipeline (Part 6 / Appendix §4).

Six-phase meta-reflective pipeline that runs E31 (Reflective Learning)
and E32 (Reflective Identity) engines against accumulated learning data
and identity stores.  Produces a ``ReflectiveModeResult`` and mutates
identity stores (conclusions reinforcement/creation, journal entries).

Phases
------
  Phase 0  Input Assembly
      Load learning logs, identity stores, pending updates, and the
      optional ReflectiveModeInput from Homework Mode handoff.

  Phase 1  Meta-Learning Analysis (E31)
      Run E31 on learning log history to detect recurring failures,
      mode effectiveness, subject proficiency trends, style preferences.

  Phase 2  Identity Coherence Analysis (E32)
      Run E32 on identity stores (core memories, conclusions, journal,
      pending updates) + current emotion state to produce coherence
      scoring and contradiction / fragility reports.

  Phase 3  Cross-Reference
      Correlate E31 recurring failures and meta-patterns with E32
      identity conclusions to detect learning-identity connections
      (e.g., persistent failure in a domain that contradicts a
      self-belief about competence).

  Phase 4  Identity Store Mutations
      - Reinforce conclusions that align with E31 mode effectiveness
      - Create new conclusions from E31 meta-patterns
      - Write identity journal entries (type=REFLECTION)
      - Write to CorticalReflectionLog.identity_coherence_status

  Phase 5  Output & Summary
      Build ``ReflectiveModeResult`` with full analysis + mutation stats.

NT interaction:
  The pipeline reads NT state from the neurochem engine (if available)
  and passes it to E31 and E32 via ``update_neurochem_state()``.  The
  pipeline itself does NOT inject NT signals — it is observational.

Triggered by ``/reflective`` command.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from zados.core.types import (
    LearningLogEntry,
    ReflectiveModeInput,
    ReflectiveModeResult,
    SessionState,
)

log = logging.getLogger(__name__)


class ReflectivePipeline:
    """Reflective meta-learning mode — 6-phase identity + learning analysis.

    Parameters
    ----------
    answer_pipeline : AnswerPipeline
        Held but not currently invoked (reserved for future sub-task
        delegation, e.g., re-processing a learning entry in M3 mode).
    learning_log : LearningLogPipeline
        Source of learning log entries for E31 analysis.
    memory_layer : MemoryLayer, optional
        Full namespaced memory access (identity, thoughts, knowledge).
    unsolved_buffer : UnsolvedBuffer, optional
        Active unsolved questions queue.
    neurochem_engine : NeurochemicalEngine, optional
        Live neurochemical simulation engine for NT snapshot reads.
    """

    def __init__(
        self,
        answer_pipeline: Any = None,
        learning_log: Any = None,
        memory_layer: Any = None,
        unsolved_buffer: Any = None,
        neurochem_engine: Any = None,
    ) -> None:
        self._pipeline = answer_pipeline
        self._learning_log = learning_log
        self._memory = memory_layer
        self._unsolved = unsolved_buffer
        self._neurochem = neurochem_engine

        # Engines — lazy-created on first process() call
        self._e31: Any = None
        self._e32: Any = None

    # ==================================================================
    # Engine initialisation (lazy)
    # ==================================================================

    def _ensure_engines(self) -> None:
        """Create E31 / E32 instances if not yet initialised."""
        if self._e31 is None:
            from zados.cognitive_engines.py_engines.reflective_learning_engine import (
                ReflectiveLearningEngine,
            )
            self._e31 = ReflectiveLearningEngine()

        if self._e32 is None:
            from zados.cognitive_engines.py_engines.reflective_identity_engine import (
                ReflectiveIdentityEngine,
            )
            self._e32 = ReflectiveIdentityEngine()

    # ==================================================================
    # Main entry point
    # ==================================================================

    def process(self, session: SessionState) -> Dict[str, Any]:
        """Run the full 6-phase reflective pipeline.

        Parameters
        ----------
        session : SessionState

        Returns
        -------
        Dict[str, Any]
            Serialisable summary of the reflective analysis.  The full
            ``ReflectiveModeResult`` is also stored on session state for
            downstream consumption.
        """
        log.info(
            "Reflective Pipeline activated for session %s.",
            session.session_id,
        )
        start_time = time.time()
        self._ensure_engines()

        result = ReflectiveModeResult(session_id=session.session_id)

        # --- Phase 0: Input Assembly ---
        learning_entries, identity_data, reflective_input = (
            self._phase0_input_assembly(session)
        )
        result.learning_logs_analysed = len(learning_entries)
        if reflective_input is not None:
            result.fallacy_flags_processed = len(reflective_input.fallacy_flags)
            result.bias_flags_processed = len(reflective_input.bias_flags)
            result.meta_patterns_processed = len(reflective_input.meta_patterns)

        # --- Phase 1: Meta-Learning Analysis (E31) ---
        e31_result = self._phase1_meta_learning(
            learning_entries, identity_data, session,
        )
        result.learning_patterns = e31_result.get("learning_patterns", [])
        result.recurring_failures = e31_result.get("recurring_failures", [])
        result.mode_effectiveness = e31_result.get("mode_effectiveness", {})
        result.subject_proficiencies = e31_result.get("subject_proficiencies", {})
        result.style_preferences = e31_result.get("style_preferences", [])
        result.learning_recommendations = e31_result.get("recommendations", [])

        # --- Phase 2: Identity Coherence Analysis (E32) ---
        e32_result = self._phase2_identity_coherence(
            identity_data, session,
        )
        result.identity_coherence_status = e32_result.get(
            "identity_coherence_status", "coherent",
        )
        result.coherence_score = e32_result.get("coherence_score", 1.0)
        result.core_contradictions = e32_result.get(
            "identity_contradictions", [],
        )
        result.fragile_conclusions = e32_result.get("fragile_conclusions", [])
        alignment_data = e32_result.get("alignment_analysis", {})
        result.alignment_issues = (
            alignment_data.get("alignment_issues", [])
            if isinstance(alignment_data, dict) else []
        )
        result.identity_themes = e32_result.get("identity_themes", [])
        pending_analysis = e32_result.get("pending_update_analysis", {})
        result.pending_updates_analysed = (
            pending_analysis.get("total", 0)
            if isinstance(pending_analysis, dict) else 0
        )

        # --- Phase 3: Cross-Reference (E31 × E32) ---
        cross_refs = self._phase3_cross_reference(
            e31_result, e32_result, reflective_input,
        )
        result.cross_references = cross_refs

        # --- Phase 4: Identity Store Mutations ---
        mutation_stats = self._phase4_identity_mutations(
            e31_result, e32_result, cross_refs, session,
        )
        result.conclusions_reinforced = mutation_stats.get(
            "conclusions_reinforced", 0,
        )
        result.conclusions_created = mutation_stats.get(
            "conclusions_created", 0,
        )
        result.conclusions_recommended_for_update = mutation_stats.get(
            "conclusions_recommended_for_update", 0,
        )
        result.journal_entries_created = mutation_stats.get(
            "journal_entries_created", 0,
        )

        # --- Phase 5: Output & Summary ---
        output = self._phase5_output(result, start_time)

        log.info(
            "Reflective Pipeline completed: coherence=%s (%.2f), "
            "patterns=%d, failures=%d, journal_entries=%d, "
            "conclusions_reinforced=%d, conclusions_created=%d "
            "(%.1fs).",
            result.identity_coherence_status,
            result.coherence_score,
            len(result.learning_patterns),
            len(result.recurring_failures),
            result.journal_entries_created,
            result.conclusions_reinforced,
            result.conclusions_created,
            time.time() - start_time,
        )

        return output

    # ==================================================================
    # Phase 0 — Input Assembly
    # ==================================================================

    def _phase0_input_assembly(
        self,
        session: SessionState,
    ) -> tuple:
        """Gather inputs from learning logs, identity stores, and homework handoff.

        Returns
        -------
        (learning_entries, identity_data, reflective_input)
            learning_entries : List[Dict[str, Any]]
                Serialised learning log entries for E31.
            identity_data : Dict[str, Any]
                Core memories, conclusions, journal, pending updates for E32.
            reflective_input : ReflectiveModeInput or None
                Homework Mode handoff (if available).
        """
        # 0.1: Gather learning logs
        learning_entries: List[Dict[str, Any]] = []
        if self._learning_log is not None:
            raw_entries: List[LearningLogEntry] = []
            try:
                raw_entries = self._learning_log.get_unprocessed_logs()
            except Exception:
                pass
            if not raw_entries:
                # Fall back to all entries (reflective mode analyses everything)
                try:
                    raw_entries = getattr(self._learning_log, "_entries", [])
                except Exception:
                    pass

            for entry in raw_entries:
                learning_entries.append(self._serialise_log_entry(entry))

        log.debug(
            "Phase 0: Gathered %d learning log entries.", len(learning_entries),
        )

        # 0.2: Gather identity data from memory layer
        identity_data = self._gather_identity_data()

        # 0.3: Check for Homework Mode handoff
        reflective_input = self._load_reflective_input(session)

        return learning_entries, identity_data, reflective_input

    def _gather_identity_data(self) -> Dict[str, Any]:
        """Read all identity sub-stores into a dict for E32."""
        data: Dict[str, Any] = {
            "core_memories": [],
            "conclusions": [],
            "journal_entries": [],
            "pending_updates": [],
            "hardcoded": [],
        }

        if self._memory is None:
            return data

        identity = getattr(self._memory, "identity", None)
        if identity is None:
            return data

        # Core memories
        core_store = getattr(identity, "core", None)
        if core_store is not None:
            try:
                for mem in core_store.get_all():
                    data["core_memories"].append({
                        "memory_id": mem.memory_id,
                        "content": mem.content,
                        "memory_type": mem.memory_type,
                        "tags": list(mem.tags),
                        "version": mem.version,
                        "update_count": len(mem.update_history),
                    })
            except Exception as e:
                log.debug("Failed to read core memories: %s", e)

        # Identity conclusions
        conclusion_store = getattr(identity, "conclusions", None)
        if conclusion_store is not None:
            try:
                for c in conclusion_store.get_all():
                    data["conclusions"].append({
                        "conclusion_id": c.conclusion_id,
                        "content": c.content,
                        "conclusion_type": c.conclusion_type,
                        "confidence": c.confidence,
                        "reinforcement_count": c.reinforcement_count,
                        "tags": list(c.tags),
                        "source_refs": list(c.source_refs),
                    })
            except Exception as e:
                log.debug("Failed to read conclusions: %s", e)

        # Identity journal (recent only — last 50 entries)
        journal_store = getattr(identity, "journal", None)
        if journal_store is not None:
            try:
                entries = journal_store.get_all()
                # Sort by timestamp descending, take latest 50
                entries_sorted = sorted(
                    entries,
                    key=lambda e: e.timestamp,
                    reverse=True,
                )[:50]
                for e in entries_sorted:
                    data["journal_entries"].append({
                        "entry_id": e.entry_id,
                        "entry_type": e.entry_type.value if hasattr(e.entry_type, "value") else str(e.entry_type),
                        "content": e.content,
                        "emotion_tags": list(e.emotion_tags),
                        "source_pipeline": e.source_pipeline,
                        "nt_snapshot": dict(e.nt_snapshot) if e.nt_snapshot else {},
                    })
            except Exception as e:
                log.debug("Failed to read journal entries: %s", e)

        # Pending updates
        pending_store = getattr(identity, "pending", None)
        if pending_store is not None:
            try:
                for u in pending_store.get_pending():
                    data["pending_updates"].append({
                        "update_id": u.update_id,
                        "target_memory_id": u.target_memory_id,
                        "proposed_content": u.proposed_content,
                        "reason": u.reason,
                        "status": u.status,
                    })
            except Exception as e:
                log.debug("Failed to read pending updates: %s", e)

        # Hardcoded identity (read-only baseline)
        hardcoded_store = getattr(identity, "hardcoded", None)
        if hardcoded_store is not None:
            try:
                for h in hardcoded_store.get_all():
                    data["hardcoded"].append({
                        "entry_id": getattr(h, "entry_id", ""),
                        "content": getattr(h, "content", str(h)),
                        "category": getattr(h, "category", ""),
                        "tags": getattr(h, "tags", []),
                    })
            except Exception as e:
                log.debug("Failed to read hardcoded identity: %s", e)

        # Identity correlations (fixed ↔ developmental mappings)
        data["correlations"] = []
        correlation_store = getattr(identity, "correlation", None)
        if correlation_store is not None:
            try:
                for c in correlation_store.get_all():
                    data["correlations"].append({
                        "correlation_id": c.correlation_id,
                        "hardcoded_entry_id": c.hardcoded_entry_id,
                        "developmental_id": c.developmental_id,
                        "developmental_type": c.developmental_type,
                        "relation_type": c.relation_type,
                        "description": c.description,
                        "confidence": c.confidence,
                        "validation_count": c.validation_count,
                    })
            except Exception as e:
                log.debug("Failed to read identity correlations: %s", e)

        log.debug(
            "Phase 0: Identity data — %d core, %d conclusions, "
            "%d journal, %d pending, %d hardcoded, %d correlations.",
            len(data["core_memories"]),
            len(data["conclusions"]),
            len(data["journal_entries"]),
            len(data["pending_updates"]),
            len(data["hardcoded"]),
            len(data["correlations"]),
        )
        return data

    def _load_reflective_input(
        self,
        session: SessionState,
    ) -> Optional[ReflectiveModeInput]:
        """Load Homework Mode handoff from session if available.

        The HomeworkPipeline stores a ReflectiveModeInput on the session
        (or in the STMM) when it detects fallacy/bias patterns.
        """
        # Check session for stored handoff
        handoff = getattr(session, "reflective_input", None)
        if isinstance(handoff, ReflectiveModeInput):
            log.debug(
                "Phase 0: Found homework handoff — %d fallacy, %d bias, "
                "%d meta-patterns from session %s.",
                len(handoff.fallacy_flags),
                len(handoff.bias_flags),
                len(handoff.meta_patterns),
                handoff.source_homework_session,
            )
            return handoff

        # Check memory layer for persisted handoff (future persistence path)
        if self._memory is not None:
            thoughts = getattr(self._memory, "thoughts", None)
            if thoughts is not None:
                overview = getattr(thoughts, "overview_logs", None)
                if overview is not None:
                    try:
                        # Search for latest homework handoff marker
                        results = overview.search("homework reflective handoff", limit=1)
                        if results:
                            log.debug("Phase 0: Found persisted homework handoff marker.")
                    except Exception:
                        pass

        return None

    # ==================================================================
    # Phase 1 — Meta-Learning Analysis (E31)
    # ==================================================================

    def _phase1_meta_learning(
        self,
        learning_entries: List[Dict[str, Any]],
        identity_data: Dict[str, Any],
        session: SessionState,
    ) -> Dict[str, Any]:
        """Run E31 Reflective Learning Engine.

        Parameters
        ----------
        learning_entries : List[Dict]
            Serialised learning log entries.
        identity_data : Dict
            Identity store data (core memories for context).
        session : SessionState

        Returns
        -------
        Dict[str, Any]
            E31 analysis results.
        """
        if self._e31 is None:
            return {}

        # Feed NT state to E31
        nt_snapshot = self._read_nt_snapshot()
        if nt_snapshot:
            self._e31.update_neurochem_state(nt_snapshot)

        # Build identity context for E31 (themes from core memories)
        identity_context: Dict[str, Any] = {}
        for mem in identity_data.get("core_memories", []):
            mem_type = mem.get("memory_type", "unknown")
            if mem_type not in identity_context:
                identity_context[mem_type] = []
            identity_context[mem_type].append(mem.get("content", "")[:200])

        try:
            result = self._e31.process(
                learning_entries=learning_entries,
                identity_context=identity_context,
            )
            log.debug(
                "Phase 1 (E31): %d patterns, %d failures, %d recommendations.",
                len(result.get("learning_patterns", [])),
                len(result.get("recurring_failures", [])),
                len(result.get("recommendations", [])),
            )
            return result
        except Exception as e:
            log.warning("Phase 1 (E31) failed: %s", e)
            return {}

    # ==================================================================
    # Phase 2 — Identity Coherence Analysis (E32)
    # ==================================================================

    def _phase2_identity_coherence(
        self,
        identity_data: Dict[str, Any],
        session: SessionState,
    ) -> Dict[str, Any]:
        """Run E32 Reflective Identity Engine.

        Parameters
        ----------
        identity_data : Dict
            Core memories, conclusions, journal entries, pending updates.
        session : SessionState

        Returns
        -------
        Dict[str, Any]
            E32 coherence analysis results.
        """
        if self._e32 is None:
            return {}

        # Feed NT state to E32
        nt_snapshot = self._read_nt_snapshot()
        if nt_snapshot:
            self._e32.update_neurochem_state(nt_snapshot)

        # Read current emotion state from session STMM if available
        emotion_state: Dict[str, float] = {}
        if self._memory is not None:
            stmm = getattr(self._memory, "stmm", None)
            if stmm is not None:
                emotion_profile = getattr(stmm, "emotion_profile", None)
                if emotion_profile and isinstance(emotion_profile, dict):
                    emotion_state = dict(emotion_profile)

        try:
            result = self._e32.process(
                core_memories=identity_data.get("core_memories", []),
                identity_conclusions=identity_data.get("conclusions", []),
                journal_entries=identity_data.get("journal_entries", []),
                pending_updates=identity_data.get("pending_updates", []),
                emotion_snapshot=emotion_state,
                hardcoded_values=identity_data.get("hardcoded", []),
            )
            pending_analysis = result.get("pending_update_analysis", {})
            pending_total = (
                pending_analysis.get("total", 0)
                if isinstance(pending_analysis, dict) else 0
            )
            log.debug(
                "Phase 2 (E32): coherence=%s (%.2f), %d contradictions, "
                "%d fragile conclusions, %d pending analysed.",
                result.get("identity_coherence_status", "coherent"),
                result.get("coherence_score", 1.0),
                len(result.get("identity_contradictions", [])),
                len(result.get("fragile_conclusions", [])),
                pending_total,
            )
            return result
        except Exception as e:
            log.warning("Phase 2 (E32) failed: %s", e)
            return {}

    # ==================================================================
    # Phase 3 — Cross-Reference (E31 × E32)
    # ==================================================================

    def _phase3_cross_reference(
        self,
        e31_result: Dict[str, Any],
        e32_result: Dict[str, Any],
        reflective_input: Optional[ReflectiveModeInput],
    ) -> List[Dict[str, Any]]:
        """Cross-reference learning failures with identity conclusions.

        Detects cases where:
          - A recurring learning failure contradicts a self-belief
            (e.g., "I'm good at logic" but E31 shows logic failures)
          - A fragile conclusion is undermined by meta-patterns
          - Homework fallacy/bias flags relate to identity themes

        Returns
        -------
        List[Dict[str, Any]]
            Cross-reference entries.
        """
        cross_refs: List[Dict[str, Any]] = []

        recurring_failures = e31_result.get("recurring_failures", [])
        contradictions = e32_result.get("identity_contradictions", [])
        fragile_conclusions = e32_result.get("fragile_conclusions", [])
        subject_proficiencies = e31_result.get("subject_proficiencies", {})
        raw_themes = e32_result.get("identity_themes", [])
        # Normalise themes to strings (E32 returns list of dicts)
        themes = [
            t.get("theme", "") if isinstance(t, dict) else str(t)
            for t in raw_themes
        ]

        # 3.1: Check if any recurring failure subject maps to an identity theme
        for failure in recurring_failures:
            failure_type = failure.get("failure_type", "")
            failure_subject = failure.get("subject", "")
            failure_count = failure.get("occurrences", 0)
            failure_words = set(failure_type.lower().split())

            for conclusion in fragile_conclusions:
                conclusion_text = conclusion.get("content", "").lower()
                conclusion_words = set(conclusion_text.split())

                # Check for semantic overlap (failure domain vs conclusion claim)
                overlap = failure_words & conclusion_words
                if len(overlap) >= 2:
                    cross_refs.append({
                        "type": "failure_vs_identity",
                        "failure_type": failure_type,
                        "failure_count": failure_count,
                        "conclusion_id": conclusion.get("conclusion_id", ""),
                        "conclusion_content": conclusion.get("content", "")[:200],
                        "overlap_terms": list(overlap),
                        "severity": "high" if failure_count >= 3 else "medium",
                    })

        # 3.2: Check subject proficiency vs identity themes
        for subject, subj_data in subject_proficiencies.items():
            trend = (
                subj_data.get("trend", "")
                if isinstance(subj_data, dict) else str(subj_data)
            )
            if trend in ("stagnating", "declining"):
                subject_lower = subject.lower()
                for theme in themes:
                    if subject_lower in theme.lower():
                        cross_refs.append({
                            "type": "proficiency_vs_theme",
                            "subject": subject,
                            "trend": trend,
                            "theme": theme,
                            "severity": "medium",
                        })

        # 3.3: Incorporate homework handoff signals
        if reflective_input is not None:
            # Map fallacy patterns to identity contradictions
            for fallacy in reflective_input.fallacy_flags:
                fallacy_name = fallacy.get("name", "")
                for contradiction in contradictions:
                    contradiction_text = str(contradiction)
                    if fallacy_name.lower() in contradiction_text.lower():
                        cross_refs.append({
                            "type": "homework_fallacy_vs_identity",
                            "fallacy": fallacy_name,
                            "contradiction": contradiction_text[:200],
                            "severity": fallacy.get("severity", "low"),
                        })

            # Map meta-patterns to identity themes
            for pattern in reflective_input.meta_patterns:
                pattern_type = pattern.get("type", "")
                for theme in themes:
                    if pattern_type.lower() in theme.lower():
                        cross_refs.append({
                            "type": "homework_pattern_vs_theme",
                            "pattern_type": pattern_type,
                            "theme": theme,
                            "severity": "low",
                        })

        log.debug(
            "Phase 3: Generated %d cross-references.", len(cross_refs),
        )
        return cross_refs

    # ==================================================================
    # Phase 4 — Identity Store Mutations
    # ==================================================================

    def _phase4_identity_mutations(
        self,
        e31_result: Dict[str, Any],
        e32_result: Dict[str, Any],
        cross_refs: List[Dict[str, Any]],
        session: SessionState,
    ) -> Dict[str, int]:
        """Apply identity store mutations based on analysis results.

        Mutations:
          a) Reinforce conclusions that align with E31 mode effectiveness
          b) Create new conclusions from E31 meta-patterns
          c) Write identity journal entries (type=REFLECTION)
          d) Update CorticalReflectionLog.identity_coherence_status

        Returns
        -------
        Dict[str, int]
            Mutation statistics.
        """
        stats: Dict[str, int] = {
            "conclusions_reinforced": 0,
            "conclusions_created": 0,
            "conclusions_recommended_for_update": 0,
            "journal_entries_created": 0,
        }

        identity = None
        if self._memory is not None:
            identity = getattr(self._memory, "identity", None)

        # 4a: Reinforce aligned conclusions
        if identity is not None:
            conclusion_store = getattr(identity, "conclusions", None)
            if conclusion_store is not None:
                stats["conclusions_reinforced"] = self._reinforce_conclusions(
                    e31_result, e32_result, conclusion_store,
                )

        # 4b: Create new conclusions from meta-patterns
        if identity is not None:
            conclusion_store = getattr(identity, "conclusions", None)
            if conclusion_store is not None:
                stats["conclusions_created"] = self._create_conclusions(
                    e31_result, cross_refs, conclusion_store,
                )

        # 4c: Recommend conclusion updates (from E32)
        conclusion_updates = e32_result.get("conclusion_updates", [])
        stats["conclusions_recommended_for_update"] = len(conclusion_updates)

        # 4c-b: Submit conclusion updates to PendingUpdateQueue
        if identity is not None and conclusion_updates:
            pending_store = getattr(identity, "pending", None)
            if pending_store is not None:
                try:
                    from zados.memory.long_term.identity.types import PendingUpdate
                    for upd_data in conclusion_updates:
                        if not isinstance(upd_data, dict):
                            continue
                        target_id = upd_data.get("target_id", upd_data.get("conclusion_id", ""))
                        proposed = upd_data.get("proposed_content", "")
                        reason = upd_data.get("reason", "E32 reflective analysis")
                        if not target_id or not proposed:
                            continue
                        pending = PendingUpdate(
                            target_memory_id=target_id,
                            proposed_content=proposed,
                            reason=reason,
                        )
                        pending_store.submit(pending)
                        stats["pending_updates_submitted"] = stats.get("pending_updates_submitted", 0) + 1
                except Exception:
                    log.debug("PendingUpdateQueue submission failed.", exc_info=True)

        # 4d: Write identity journal entries
        if identity is not None:
            journal_store = getattr(identity, "journal", None)
            if journal_store is not None:
                stats["journal_entries_created"] = self._write_journal_entries(
                    e31_result, e32_result, cross_refs, journal_store, session,
                )

        # 4e: Update CorticalReflectionLog.identity_coherence_status
        coherence_status = e32_result.get("identity_coherence_status", "coherent")
        self._update_cortical_coherence(session, coherence_status)

        # 4f: Create/update identity correlations (fixed ↔ developmental)
        if identity is not None:
            correlation_store = getattr(identity, "correlation", None)
            hardcoded_store = getattr(identity, "hardcoded", None)
            conclusion_store_for_corr = getattr(identity, "conclusions", None)
            if correlation_store is not None and hardcoded_store is not None:
                stats["correlations_created"] = self._update_correlations(
                    e32_result, cross_refs,
                    correlation_store, hardcoded_store, conclusion_store_for_corr,
                )

        log.debug(
            "Phase 4: reinforced=%d, created=%d, recommended=%d, journal=%d, "
            "correlations=%d.",
            stats["conclusions_reinforced"],
            stats["conclusions_created"],
            stats["conclusions_recommended_for_update"],
            stats["journal_entries_created"],
            stats.get("correlations_created", 0),
        )
        return stats

    def _reinforce_conclusions(
        self,
        e31_result: Dict[str, Any],
        e32_result: Dict[str, Any],
        conclusion_store: Any,
    ) -> int:
        """Reinforce conclusions that have supporting evidence.

        Evidence sources:
          - E31 mode_effectiveness with high confirmation ratios
          - E32 alignment checks that passed
          - Subject proficiencies that are improving

        Returns number of conclusions reinforced.
        """
        reinforced = 0

        # Conclusions from E32 that are NOT fragile and NOT contradicted
        fragile_ids = {
            c.get("conclusion_id", "")
            for c in e32_result.get("fragile_conclusions", [])
        }
        contradicted_ids = set()
        for c in e32_result.get("identity_contradictions", []):
            for cid in (c.get("entry_a", ""), c.get("entry_b", "")):
                if cid:
                    contradicted_ids.add(cid)

        # Get all conclusions and reinforce the stable ones
        try:
            all_conclusions = conclusion_store.get_all()
        except Exception:
            return 0

        improving_subjects: set = set()
        for subj, subj_data in e31_result.get(
            "subject_proficiencies", {},
        ).items():
            trend = (
                subj_data.get("trend", "")
                if isinstance(subj_data, dict) else str(subj_data)
            )
            if trend == "improving":
                improving_subjects.add(subj)

        for c in all_conclusions:
            cid = c.conclusion_id
            if cid in fragile_ids or cid in contradicted_ids:
                continue

            # Check if this conclusion relates to an improving subject
            c_text_lower = c.content.lower()
            should_reinforce = False

            for subj in improving_subjects:
                if subj.lower() in c_text_lower:
                    should_reinforce = True
                    break

            # Also reinforce high-confidence conclusions of type "lesson"
            if c.conclusion_type == "lesson" and c.confidence >= 0.7:
                should_reinforce = True

            if should_reinforce:
                try:
                    if conclusion_store.reinforce(cid):
                        reinforced += 1
                except Exception as e:
                    log.debug("Failed to reinforce conclusion %s: %s", cid, e)

        return reinforced

    def _create_conclusions(
        self,
        e31_result: Dict[str, Any],
        cross_refs: List[Dict[str, Any]],
        conclusion_store: Any,
    ) -> int:
        """Create new identity conclusions from meta-patterns.

        Sources:
          - E31 recurring_failures → "lesson" conclusions
          - E31 style_preferences → "self_insight" conclusions
          - Cross-references (failure_vs_identity) → "self_insight"

        Returns number of conclusions created.
        """
        created = 0

        # Lazy import to avoid circular deps
        try:
            from zados.memory.long_term.identity.types import IdentityConclusion
        except ImportError:
            log.debug("Cannot import IdentityConclusion — skipping conclusion creation.")
            return 0

        # From recurring failures — create "lesson" conclusions
        for failure in e31_result.get("recurring_failures", []):
            failure_type = failure.get("failure_type", "")
            failure_count = failure.get("occurrences", 0)
            if failure_count >= 3 and failure_type:
                conclusion = IdentityConclusion(
                    content=(
                        f"Recurring learning challenge: {failure_type} "
                        f"(observed {failure_count} times across sessions). "
                        f"This pattern suggests a systematic gap that "
                        f"needs targeted practice."
                    ),
                    conclusion_type="lesson",
                    confidence=min(0.4 + (failure_count * 0.05), 0.8),
                    tags=["reflective_mode", "recurring_failure", failure_type],
                    source_refs=["E31_reflective_learning"],
                )
                try:
                    conclusion_store.write(conclusion)
                    created += 1
                except Exception as e:
                    log.debug("Failed to create failure conclusion: %s", e)

        # From style preferences — create "self_insight" conclusions
        style_prefs = e31_result.get("style_preferences", [])
        if style_prefs:
            top_pref = style_prefs[0] if style_prefs else {}
            top_mode = (
                top_pref.get("mode", "") if isinstance(top_pref, dict)
                else str(top_pref)
            )
            if top_mode:
                conclusion = IdentityConclusion(
                    content=(
                        f"Learning style insight: most effective learning "
                        f"occurs in {top_mode} mode. This suggests a "
                        f"preference for "
                        + self._describe_mode_style(top_mode)
                        + "."
                    ),
                    conclusion_type="self_insight",
                    confidence=0.5,
                    tags=["reflective_mode", "learning_style", top_mode],
                    source_refs=["E31_reflective_learning"],
                )
                try:
                    conclusion_store.write(conclusion)
                    created += 1
                except Exception as e:
                    log.debug("Failed to create style conclusion: %s", e)

        # From high-severity cross-references
        for xref in cross_refs:
            if xref.get("severity") == "high" and xref.get("type") == "failure_vs_identity":
                conclusion = IdentityConclusion(
                    content=(
                        f"Identity-learning tension: recurring failure in "
                        f"'{xref.get('failure_type', '')}' may conflict with "
                        f"self-belief: '{xref.get('conclusion_content', '')[:100]}'. "
                        f"This warrants re-evaluation."
                    ),
                    conclusion_type="self_insight",
                    confidence=0.4,
                    tags=["reflective_mode", "identity_tension"],
                    source_refs=[
                        "E31_reflective_learning",
                        "E32_reflective_identity",
                        xref.get("conclusion_id", ""),
                    ],
                )
                try:
                    conclusion_store.write(conclusion)
                    created += 1
                except Exception as e:
                    log.debug("Failed to create cross-ref conclusion: %s", e)

        return created

    def _write_journal_entries(
        self,
        e31_result: Dict[str, Any],
        e32_result: Dict[str, Any],
        cross_refs: List[Dict[str, Any]],
        journal_store: Any,
        session: SessionState,
    ) -> int:
        """Write identity journal entries summarising the reflective session.

        Creates one REFLECTION entry per major finding area.

        Returns number of entries created.
        """
        created = 0

        try:
            from zados.memory.long_term.identity.types import (
                IdentityJournalEntry,
                IdentityJournalEntryType,
            )
        except ImportError:
            log.debug("Cannot import IdentityJournalEntry — skipping journal writes.")
            return 0

        nt_snapshot = self._read_nt_snapshot()

        # Entry 1: Learning meta-analysis summary
        patterns = e31_result.get("learning_patterns", [])
        failures = e31_result.get("recurring_failures", [])
        recommendations = e31_result.get("recommendations", [])

        if patterns or failures or recommendations:
            summary_parts = []
            if patterns:
                summary_parts.append(
                    f"Detected {len(patterns)} meta-learning pattern(s)."
                )
            if failures:
                failure_types = [f.get("failure_type", "?") for f in failures[:3]]
                summary_parts.append(
                    f"Recurring challenges: {', '.join(failure_types)}."
                )
            if recommendations:
                rec_texts = [
                    r.get("recommendation", str(r))
                    if isinstance(r, dict) else str(r)
                    for r in recommendations[:3]
                ]
                summary_parts.append(
                    f"Recommendations: {'; '.join(rec_texts)}."
                )

            entry = IdentityJournalEntry(
                entry_type=IdentityJournalEntryType.REFLECTION,
                content=" ".join(summary_parts),
                nt_snapshot=nt_snapshot or {},
                emotion_tags=[],
                source_pipeline="reflective_mode",
                tags=["meta_learning", "E31_analysis", session.session_id],
            )
            try:
                journal_store.write(entry)
                created += 1
            except Exception as e:
                log.debug("Failed to write learning journal entry: %s", e)

        # Entry 2: Identity coherence summary
        coherence_status = e32_result.get("identity_coherence_status", "coherent")
        contradictions = e32_result.get("identity_contradictions", [])
        fragile = e32_result.get("fragile_conclusions", [])

        if coherence_status != "coherent" or contradictions or fragile:
            parts = [
                f"Identity coherence: {coherence_status} "
                f"(score: {e32_result.get('coherence_score', 1.0):.2f})."
            ]
            if contradictions:
                parts.append(
                    f"Detected {len(contradictions)} internal contradiction(s)."
                )
            if fragile:
                parts.append(
                    f"{len(fragile)} conclusion(s) flagged as fragile "
                    f"(low confidence or reinforcement)."
                )

            # Tag with identity-relevant emotions from the E32 analysis
            emotion_tags = []
            identity_emotions = e32_result.get("identity_emotions", {})
            active_emotions = (
                identity_emotions.get("active_identity_emotions", {})
                if isinstance(identity_emotions, dict) else {}
            )
            for emo, score in active_emotions.items():
                if isinstance(score, (int, float)) and score > 0.3:
                    emotion_tags.append(emo)

            entry = IdentityJournalEntry(
                entry_type=IdentityJournalEntryType.REFLECTION,
                content=" ".join(parts),
                nt_snapshot=nt_snapshot or {},
                emotion_tags=emotion_tags,
                source_pipeline="reflective_mode",
                tags=["identity_coherence", "E32_analysis", session.session_id],
            )
            try:
                journal_store.write(entry)
                created += 1
            except Exception as e:
                log.debug("Failed to write coherence journal entry: %s", e)

        # Entry 3: Cross-reference findings (if any high severity)
        high_xrefs = [x for x in cross_refs if x.get("severity") == "high"]
        if high_xrefs:
            xref_parts = [
                f"Identity-learning tension detected ({len(high_xrefs)} "
                f"high-severity cross-reference(s)):"
            ]
            for xref in high_xrefs[:3]:
                xref_parts.append(
                    f"  - {xref.get('type', '')}: {xref.get('failure_type', xref.get('subject', ''))}"
                )

            entry = IdentityJournalEntry(
                entry_type=IdentityJournalEntryType.REFLECTION,
                content=" ".join(xref_parts),
                nt_snapshot=nt_snapshot or {},
                emotion_tags=[],
                source_pipeline="reflective_mode",
                tags=["cross_reference", "identity_tension", session.session_id],
            )
            try:
                journal_store.write(entry)
                created += 1
            except Exception as e:
                log.debug("Failed to write cross-ref journal entry: %s", e)

        return created

    def _update_cortical_coherence(
        self,
        session: SessionState,
        coherence_status: str,
    ) -> None:
        """Write identity_coherence_status to CorticalReflectionLog.

        This is the E32 → CorticalReflectionLog write path specified
        in the Appendix spec (open item: E32 write path).
        """
        if self._memory is None:
            return

        stmm = getattr(self._memory, "stmm", None)
        if stmm is None:
            return

        cortical_log = getattr(stmm, "cortical_reflection_log", None)
        if cortical_log is None:
            return

        old_status = getattr(cortical_log, "identity_coherence_status", "coherent")
        cortical_log.identity_coherence_status = coherence_status

        if old_status != coherence_status:
            log.info(
                "Phase 4: CorticalReflectionLog.identity_coherence_status "
                "updated: %s → %s.",
                old_status,
                coherence_status,
            )
            # Also add a note to the cortical log
            notes = getattr(cortical_log, "notes", [])
            notes.append(
                f"[reflective_mode] Identity coherence changed: "
                f"{old_status} → {coherence_status}"
            )

    # ==================================================================
    # Phase 4f — Identity Correlation Updates
    # ==================================================================

    def _update_correlations(
        self,
        e32_result: Dict[str, Any],
        cross_refs: List[Dict[str, Any]],
        correlation_store: Any,
        hardcoded_store: Any,
        conclusion_store: Any,
    ) -> int:
        """Create identity correlations between hardcoded and developmental entries.

        Scans E32 alignment analysis and cross-references for connections
        between fixed identity (hardcoded entries) and developmental identity
        (conclusions, core memories).  Creates IdentityCorrelation records
        for each detected relationship.

        Returns number of correlations created.
        """
        created = 0

        try:
            from zados.memory.long_term.identity.types import IdentityCorrelation
        except ImportError:
            return 0

        if conclusion_store is None:
            return 0

        # Get all hardcoded entries for keyword matching
        all_hardcoded = hardcoded_store.get_all()
        if not all_hardcoded:
            return 0

        # Get all conclusions
        try:
            all_conclusions = conclusion_store.get_all()
        except Exception:
            return 0

        # Match conclusions against hardcoded entries by tag/keyword overlap
        for conclusion in all_conclusions:
            c_text = conclusion.content.lower()
            c_tags = set(conclusion.tags)

            for hc_entry in all_hardcoded:
                hc_tags = set(hc_entry.tags)
                hc_text = hc_entry.content.lower()
                hc_id = hc_entry.entry_id

                # Check if correlation already exists
                existing = correlation_store.get_by_hardcoded(hc_id)
                already_linked = any(
                    c.developmental_id == conclusion.conclusion_id
                    for c in existing
                )
                if already_linked:
                    # Re-validate existing correlation
                    for c in existing:
                        if c.developmental_id == conclusion.conclusion_id:
                            correlation_store.validate(c.correlation_id)
                    continue

                # Determine relation by overlap
                tag_overlap = c_tags & hc_tags
                if len(tag_overlap) < 2:
                    # Also check keyword overlap in content
                    c_words = set(c_text.split())
                    hc_words = set(hc_text.split())
                    # Remove common stop words
                    stop = {"the", "a", "an", "is", "are", "was", "were", "be",
                            "been", "being", "have", "has", "had", "do", "does",
                            "did", "will", "would", "could", "should", "may",
                            "might", "shall", "can", "need", "dare", "ought",
                            "used", "to", "of", "in", "for", "on", "with", "at",
                            "by", "from", "as", "into", "through", "during",
                            "before", "after", "above", "below", "between",
                            "out", "off", "over", "under", "again", "further",
                            "then", "once", "and", "but", "or", "nor", "not",
                            "so", "yet", "both", "either", "neither", "each",
                            "every", "all", "any", "few", "more", "most",
                            "other", "some", "such", "no", "only", "own",
                            "same", "than", "too", "very", "just", "because",
                            "it", "its", "my", "i", "me", "we", "our", "that",
                            "this", "these", "those", "what", "which", "who",
                            "whom", "whose", "when", "where", "how", "why"}
                    c_words -= stop
                    hc_words -= stop
                    word_overlap = c_words & hc_words
                    if len(word_overlap) < 3:
                        continue
                    overlap_terms = word_overlap
                else:
                    overlap_terms = tag_overlap

                # Determine relation type
                relation_type = self._infer_relation_type(
                    conclusion, hc_entry, overlap_terms,
                )

                corr = IdentityCorrelation(
                    hardcoded_entry_id=hc_id,
                    developmental_id=conclusion.conclusion_id,
                    developmental_type="conclusion",
                    relation_type=relation_type,
                    description=(
                        f"Conclusion '{conclusion.content[:80]}' "
                        f"{relation_type} hardcoded '{hc_id}' "
                        f"(overlap: {', '.join(list(overlap_terms)[:5])})"
                    ),
                    confidence=min(0.5 + len(overlap_terms) * 0.05, 0.9),
                    tags=list(overlap_terms)[:10],
                )
                try:
                    correlation_store.write(corr)
                    created += 1
                except Exception:
                    log.debug("Failed to write correlation for %s ↔ %s",
                              hc_id, conclusion.conclusion_id)

        # From cross-references: identity tensions → tensions_with correlations
        for xref in cross_refs:
            if xref.get("type") == "failure_vs_identity" and xref.get("conclusion_id"):
                conclusion_id = xref["conclusion_id"]
                # Find best matching hardcoded entry by failure type
                failure_type = xref.get("failure_type", "")
                best_hc = None
                for hc in all_hardcoded:
                    if failure_type.lower() in hc.content.lower():
                        best_hc = hc
                        break
                if best_hc is None:
                    continue

                corr = IdentityCorrelation(
                    hardcoded_entry_id=best_hc.entry_id,
                    developmental_id=conclusion_id,
                    developmental_type="conclusion",
                    relation_type="tensions_with",
                    description=(
                        f"Learning failure '{failure_type}' creates tension "
                        f"between developmental conclusion and hardcoded "
                        f"'{best_hc.entry_id}'"
                    ),
                    confidence=0.6 if xref.get("severity") == "high" else 0.4,
                    tags=["reflective_mode", "identity_tension", failure_type],
                )
                try:
                    correlation_store.write(corr)
                    created += 1
                except Exception:
                    pass

        log.debug("Phase 4f: Created %d identity correlations.", created)
        return created

    @staticmethod
    def _infer_relation_type(
        conclusion: Any,
        hc_entry: Any,
        overlap_terms: set,
    ) -> str:
        """Infer the relation type between a conclusion and hardcoded entry."""
        c_type = getattr(conclusion, "conclusion_type", "")
        hc_category = getattr(hc_entry, "category", "")

        if c_type == "lesson":
            return "extends"
        if c_type == "self_insight":
            return "deepens"
        if c_type == "value" and hc_category in ("core_value", "value", "axiom"):
            return "instantiates"
        if c_type == "boundary":
            return "supports"
        if "tension" in " ".join(overlap_terms).lower():
            return "tensions_with"
        return "supports"

    # ==================================================================
    # Phase 5 — Output & Summary
    # ==================================================================

    def _phase5_output(
        self,
        result: ReflectiveModeResult,
        start_time: float,
    ) -> Dict[str, Any]:
        """Build the final output dict from the result dataclass.

        Returns
        -------
        Dict[str, Any]
            Serialisable summary.
        """
        processing_time = round(time.time() - start_time, 2)

        return {
            "status": "completed",
            "session_id": result.session_id,
            "processing_time_s": processing_time,

            # E31 — meta-learning
            "learning_patterns": len(result.learning_patterns),
            "recurring_failures": len(result.recurring_failures),
            "mode_effectiveness": result.mode_effectiveness,
            "subject_proficiencies": result.subject_proficiencies,
            "style_preferences": result.style_preferences,
            "learning_recommendations": result.learning_recommendations,

            # E32 — identity coherence
            "identity_coherence_status": result.identity_coherence_status,
            "coherence_score": result.coherence_score,
            "core_contradictions": len(result.core_contradictions),
            "fragile_conclusions": len(result.fragile_conclusions),
            "alignment_issues": len(result.alignment_issues),
            "identity_themes": result.identity_themes,

            # Cross-reference
            "cross_references": len(result.cross_references),

            # Mutations
            "conclusions_reinforced": result.conclusions_reinforced,
            "conclusions_created": result.conclusions_created,
            "conclusions_recommended_for_update": result.conclusions_recommended_for_update,
            "journal_entries_created": result.journal_entries_created,
            "pending_updates_analysed": result.pending_updates_analysed,

            # Input stats
            "learning_logs_analysed": result.learning_logs_analysed,
            "fallacy_flags_processed": result.fallacy_flags_processed,
            "bias_flags_processed": result.bias_flags_processed,
            "meta_patterns_processed": result.meta_patterns_processed,
        }

    # ==================================================================
    # Helpers
    # ==================================================================

    def _read_nt_snapshot(self) -> Optional[Dict[str, float]]:
        """Read current NT concentrations from neurochem engine.

        Returns None if neurochem not available.
        """
        if self._neurochem is None:
            return None

        try:
            state = self._neurochem.get_state()
            if state is None:
                return None
            # Extract NT concentrations as lowercase keys
            nt_map: Dict[str, float] = {}
            concentrations = getattr(state, "concentrations", None)
            if concentrations and isinstance(concentrations, dict):
                from zados.cognitive_engines.constants import normalize_nt_key
                for key, val in concentrations.items():
                    canonical = normalize_nt_key(key, target="lower")
                    nt_map[canonical] = float(val)
            return nt_map if nt_map else None
        except Exception:
            return None

    @staticmethod
    def _serialise_log_entry(entry: LearningLogEntry) -> Dict[str, Any]:
        """Convert a LearningLogEntry to a dict for E31 consumption."""
        return {
            "turn_id": entry.turn_id,
            "timestamp": entry.timestamp,
            "mode": entry.mode,
            "subject": entry.subject,
            "session_id": entry.session_id,
            "confirmations": entry.confirmations,
            "contradictions": entry.contradictions,
            "extensions": entry.extensions,
            "novel_entries": entry.novel_entries,
            "patterns_detected": entry.patterns_detected,
            "e19_patterns": entry.e19_patterns,
            "e20_comparisons": entry.e20_comparisons,
            "e17_rewards": entry.e17_rewards,
            "e25_meta_updates": entry.e25_meta_updates,
            "reward_scores": entry.reward_scores,
            "processed": entry.processed,
        }

    @staticmethod
    def _describe_mode_style(mode: str) -> str:
        """Human-readable description of a learning mode style."""
        descriptions = {
            "M1": "guided instruction (human teaches)",
            "M2": "critical peer review and error detection",
            "M3": "collaborative exploration and dialectic",
            "M4": "self-generated questioning from learned material",
            "M5": "independent study and autonomous exploration",
        }
        return descriptions.get(mode, f"the {mode} learning approach")
