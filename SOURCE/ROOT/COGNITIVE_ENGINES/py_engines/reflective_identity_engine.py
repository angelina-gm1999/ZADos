"""
ZA-DOS v0.6 — E32: Reflective Identity Engine (Appendix §4.2).

Identity coherence engine that evaluates the consistency and stability
of the self-model by analysing core memories, identity conclusions,
identity journal entries, and pending updates.

Key responsibilities:
  1. Core memory consistency check — detect internal contradictions
     between stated beliefs, values, and self-model entries
  2. Conclusion stability analysis — flag low-confidence, low-
     reinforcement conclusions that may need revision or pruning
  3. Identity-behaviour alignment — compare recent journal entries
     (which reflect actual behaviour) against core memories (which
     reflect stated values)
  4. Coherence scoring — produce a composite identity_coherence_status
     for the CorticalReflectionLog

NT coupling:
  - OXT modulates social identity sensitivity (high OXT → relational
    memories weighted more heavily in coherence scoring)
  - 5-HT modulates stability bias (high 5-HT → tolerates more
    divergence before flagging disruption)
  - DA modulates self-relevance salience (high DA → focuses on
    identity-relevant patterns over neutral ones)
  - COR/CRH modulates threat sensitivity (high COR → lower threshold
    for identity_coherence_status = "disrupted")

Appendix cross-ref:
  - Confused > 0.6 → identity_coherence_status = "disrupted"
  - Ashamed / Guilty → LTMM self-model correction
  - Proud / Belonging → identity reinforcement
  - Accepted → identity-performance coherence validation
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from zados.cognitive_engines.constants import _clamp

log = logging.getLogger(__name__)

# Emotions that, when detected, indicate identity-relevant processing
IDENTITY_RELEVANT_EMOTIONS = frozenset({
    "ashamed", "guilty", "regret", "proud", "belonging",
    "accepted", "respected", "rejected", "betrayal",
    "isolated", "grief", "numb", "connected", "loyal",
    "valued", "sensitive",
})

# Coherence status levels (ordered from stable to disrupted)
COHERENCE_COHERENT = "coherent"
COHERENCE_FRAGMENTED = "fragmented"
COHERENCE_DISRUPTED = "disrupted"


class ReflectiveIdentityEngine:
    """Engine 32 — Reflective Identity: identity coherence analysis."""

    engine_id = 32
    engine_name = "reflective_identity_engine"
    cluster = "metacognition"

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------
    _CONTRADICTION_SIMILARITY_THRESHOLD = 0.3   # word overlap for contradiction detection
    _LOW_CONFIDENCE_THRESHOLD = 0.3             # conclusions below this are fragile
    _LOW_REINFORCEMENT_THRESHOLD = 2            # conclusions reinforced fewer times
    _COHERENCE_DISRUPTION_THRESHOLD = 0.4       # composite score below this = disrupted
    _COHERENCE_FRAGMENTED_THRESHOLD = 0.7       # composite score below this = fragmented

    def __init__(self) -> None:
        self._nt_state: Dict[str, float] = {}
        self._active = True

    # ------------------------------------------------------------------
    # Neurochem interface
    # ------------------------------------------------------------------

    def update_neurochem_state(self, nt_state: Dict[str, float]) -> None:
        """Accept latest NT state snapshot.

        Parameters
        ----------
        nt_state : Dict[str, float]
            Canonical lowercase keys (da, 5ht, ne, ach, oxt, cor, etc.).
        """
        self._nt_state = dict(nt_state)

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(
        self,
        *,
        core_memories: Optional[List[Dict[str, Any]]] = None,
        identity_conclusions: Optional[List[Dict[str, Any]]] = None,
        journal_entries: Optional[List[Dict[str, Any]]] = None,
        pending_updates: Optional[List[Dict[str, Any]]] = None,
        emotion_snapshot: Optional[Dict[str, float]] = None,
        hardcoded_values: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Analyse identity stores for coherence and consistency.

        Parameters
        ----------
        core_memories : list of dict
            Serialised CoreMemory entries (content, memory_type, tags, version).
        identity_conclusions : list of dict
            Serialised IdentityConclusion entries (content, conclusion_type,
            confidence, reinforcement_count, source_refs).
        journal_entries : list of dict
            Serialised IdentityJournalEntry entries (content, entry_type,
            emotion_tags, nt_snapshot, source_pipeline).
        pending_updates : list of dict
            Serialised PendingUpdate entries awaiting approval.
        emotion_snapshot : dict
            Current emotion_profile (emotion_name → intensity).
        hardcoded_values : list of dict
            Immutable baseline identity entries (axioms, values, constraints).

        Returns
        -------
        Dict[str, Any]
        """
        cores = core_memories or []
        conclusions = identity_conclusions or []
        journals = journal_entries or []
        pending = pending_updates or []
        emotions = emotion_snapshot or {}
        hardcoded = hardcoded_values or []

        if not cores and not conclusions and not journals:
            log.debug("E32: No identity data to analyse.")
            return self._empty_result()

        # NT modulation factors
        oxt = _clamp(self._nt_state.get("oxt", 0.5))
        sht = _clamp(self._nt_state.get("5ht", 0.5))
        da = _clamp(self._nt_state.get("da", 0.5))
        cor = _clamp(self._nt_state.get("cor", 0.0))

        # Social identity weight: high OXT → relational memories matter more
        social_weight = 0.3 + 0.4 * oxt    # range [0.3, 0.7]

        # Stability tolerance: high 5-HT → more divergence tolerated
        stability_tolerance = 0.2 + 0.3 * sht  # range [0.2, 0.5]

        # Self-relevance salience: high DA → stricter identity checks
        self_relevance = 0.5 + 0.3 * da    # range [0.5, 0.8]

        # Threat sensitivity: high COR → lower disruption threshold
        threat_bias = 0.0 + 0.15 * cor     # range [0.0, 0.15]

        # ---- Analysis passes ----
        contradictions = self._detect_core_contradictions(cores, hardcoded)
        fragile_conclusions = self._detect_fragile_conclusions(conclusions)
        alignment = self._check_identity_behaviour_alignment(
            cores, journals, social_weight,
        )
        identity_emotions = self._assess_identity_emotions(emotions)
        pending_analysis = self._analyse_pending_updates(pending, cores)

        # ---- Coherence scoring ----
        coherence_score = self._compute_coherence_score(
            cores=cores,
            contradictions=contradictions,
            fragile_conclusions=fragile_conclusions,
            alignment=alignment,
            identity_emotions=identity_emotions,
            stability_tolerance=stability_tolerance,
            threat_bias=threat_bias,
        )

        # Determine status
        adjusted_disruption = max(
            0.1,
            self._COHERENCE_DISRUPTION_THRESHOLD - threat_bias,
        )
        adjusted_fragmented = max(
            adjusted_disruption + 0.1,
            self._COHERENCE_FRAGMENTED_THRESHOLD - threat_bias,
        )

        if coherence_score < adjusted_disruption:
            coherence_status = COHERENCE_DISRUPTED
        elif coherence_score < adjusted_fragmented:
            coherence_status = COHERENCE_FRAGMENTED
        else:
            coherence_status = COHERENCE_COHERENT

        # Forced disruption: confused > 0.6 (Appendix spec)
        if emotions.get("confused", 0.0) > 0.6:
            coherence_status = COHERENCE_DISRUPTED
            log.info("E32: Forced disrupted status — confused > 0.6.")

        # Identify themes from core memories
        themes = self._extract_identity_themes(cores, conclusions)

        # Conclusion update recommendations
        conclusion_updates = self._recommend_conclusion_updates(
            conclusions, fragile_conclusions, contradictions,
            self_relevance,
        )

        result = {
            "engine_id": self.engine_name,
            "coherence_score": round(coherence_score, 3),
            "identity_coherence_status": coherence_status,
            "identity_contradictions": contradictions,
            "fragile_conclusions": fragile_conclusions,
            "alignment_analysis": alignment,
            "identity_emotions": identity_emotions,
            "pending_update_analysis": pending_analysis,
            "identity_themes": themes,
            "conclusion_updates": conclusion_updates,
            "cores_analysed": len(cores),
            "conclusions_analysed": len(conclusions),
            "journals_analysed": len(journals),
            "nt_modulation": {
                "social_weight": round(social_weight, 3),
                "stability_tolerance": round(stability_tolerance, 3),
                "self_relevance": round(self_relevance, 3),
                "threat_bias": round(threat_bias, 3),
            },
        }

        log.info(
            "E32: Identity coherence=%s (score=%.3f) — "
            "%d contradictions, %d fragile conclusions, %d pending.",
            coherence_status, coherence_score,
            len(contradictions), len(fragile_conclusions), len(pending),
        )
        return result

    # ------------------------------------------------------------------
    # Core memory contradiction detection
    # ------------------------------------------------------------------

    def _detect_core_contradictions(
        self,
        cores: List[Dict[str, Any]],
        hardcoded: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect contradictions between core memory entries.

        Uses word overlap heuristic to find pairs of memories that
        share vocabulary but express opposing sentiments (presence of
        negation words, antonym patterns).
        """
        contradictions: List[Dict[str, Any]] = []

        # Combine core memories with hardcoded for cross-checking
        all_entries = [
            {"id": c.get("memory_id", f"core_{i}"), "content": c.get("content", ""),
             "source": "core_memory", "tags": c.get("tags", [])}
            for i, c in enumerate(cores)
        ]
        for i, h in enumerate(hardcoded):
            all_entries.append({
                "id": h.get("entry_id", f"hardcoded_{i}"),
                "content": h.get("content", ""),
                "source": "hardcoded",
                "tags": h.get("tags", []),
            })

        # Pairwise comparison (O(n²) — acceptable for identity store sizes)
        negation_words = {"not", "never", "no", "don't", "doesn't", "can't",
                          "won't", "shouldn't", "isn't", "aren't", "wasn't",
                          "weren't", "unlike", "opposite", "against", "reject"}

        for i, a in enumerate(all_entries):
            a_words = set(a["content"].lower().split())
            for j, b in enumerate(all_entries):
                if j <= i:
                    continue
                b_words = set(b["content"].lower().split())

                # Compute word overlap (excluding common stop words)
                overlap = a_words & b_words
                meaningful_overlap = overlap - {
                    "the", "a", "an", "is", "are", "was", "were", "be",
                    "to", "of", "and", "in", "that", "it", "for", "on",
                    "with", "as", "at", "by", "from", "or", "i",
                }

                if not meaningful_overlap:
                    continue

                overlap_ratio = len(meaningful_overlap) / max(
                    len(a_words | b_words), 1,
                )

                if overlap_ratio < self._CONTRADICTION_SIMILARITY_THRESHOLD:
                    continue

                # Check for negation asymmetry
                a_negations = a_words & negation_words
                b_negations = b_words & negation_words

                if a_negations != b_negations and (a_negations or b_negations):
                    contradictions.append({
                        "entry_a": a["id"],
                        "entry_b": b["id"],
                        "source_a": a["source"],
                        "source_b": b["source"],
                        "overlap_words": list(meaningful_overlap)[:10],
                        "overlap_ratio": round(overlap_ratio, 3),
                        "type": "negation_asymmetry",
                        "severity": (
                            "high" if a["source"] == "hardcoded"
                            or b["source"] == "hardcoded"
                            else "medium"
                        ),
                    })

        return contradictions

    # ------------------------------------------------------------------
    # Conclusion stability analysis
    # ------------------------------------------------------------------

    def _detect_fragile_conclusions(
        self,
        conclusions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify conclusions with low confidence and few reinforcements."""
        fragile: List[Dict[str, Any]] = []

        for c in conclusions:
            confidence = c.get("confidence", 0.5)
            reinforcement = c.get("reinforcement_count", 0)
            ctype = c.get("conclusion_type", "unknown")

            if (confidence < self._LOW_CONFIDENCE_THRESHOLD
                    or reinforcement < self._LOW_REINFORCEMENT_THRESHOLD):
                fragile.append({
                    "conclusion_id": c.get("conclusion_id", ""),
                    "content": c.get("content", "")[:200],
                    "conclusion_type": ctype,
                    "confidence": confidence,
                    "reinforcement_count": reinforcement,
                    "reason": (
                        "low_confidence" if confidence < self._LOW_CONFIDENCE_THRESHOLD
                        else "low_reinforcement"
                    ),
                })

        return fragile

    # ------------------------------------------------------------------
    # Identity-behaviour alignment
    # ------------------------------------------------------------------

    def _check_identity_behaviour_alignment(
        self,
        cores: List[Dict[str, Any]],
        journals: List[Dict[str, Any]],
        social_weight: float,
    ) -> Dict[str, Any]:
        """Check alignment between core memories (values) and journal entries (actions).

        Looks for journal entries whose emotion_tags or content suggest
        behaviour that contradicts core memory values.
        """
        # Extract core value keywords
        value_cores = [
            c for c in cores
            if c.get("memory_type") in ("self_model", "event", "experience")
        ]
        core_keywords: Dict[str, List[str]] = {}
        for c in value_cores:
            mid = c.get("memory_id", "")
            words = set(c.get("content", "").lower().split())
            core_keywords[mid] = list(words)

        # Analyse recent journal entries for alignment
        alignment_issues: List[Dict[str, Any]] = []
        aligned_entries = 0
        total_checked = 0

        for j in journals[-20:]:  # check most recent 20
            total_checked += 1
            j_content = j.get("content", "").lower()
            j_emotions = set(j.get("emotion_tags", []))
            j_source = j.get("source_pipeline", "")

            # Check for identity-disruptive emotions in journal
            disruptive = j_emotions & {
                "ashamed", "guilty", "regret", "rejected", "betrayal",
            }
            reinforcing = j_emotions & {
                "proud", "belonging", "accepted", "respected", "valued",
            }

            if disruptive and not reinforcing:
                alignment_issues.append({
                    "entry_id": j.get("entry_id", ""),
                    "disruptive_emotions": list(disruptive),
                    "source": j_source,
                    "severity": "medium",
                })
            elif reinforcing:
                aligned_entries += 1

        alignment_ratio = (
            aligned_entries / total_checked if total_checked > 0 else 1.0
        )

        return {
            "entries_checked": total_checked,
            "aligned_entries": aligned_entries,
            "alignment_ratio": round(alignment_ratio, 3),
            "alignment_issues": alignment_issues,
            "social_weight_applied": round(social_weight, 3),
        }

    # ------------------------------------------------------------------
    # Identity-relevant emotion assessment
    # ------------------------------------------------------------------

    def _assess_identity_emotions(
        self,
        emotions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assess current emotion snapshot for identity-relevant signals."""
        relevant: Dict[str, float] = {}
        for emo, intensity in emotions.items():
            if emo in IDENTITY_RELEVANT_EMOTIONS and intensity > 0.0:
                relevant[emo] = round(intensity, 3)

        # Compute net valence of identity emotions
        positive = {"proud", "belonging", "accepted", "respected",
                    "valued", "connected", "loyal", "sensitive"}
        negative = {"ashamed", "guilty", "regret", "rejected",
                    "betrayal", "isolated", "grief", "numb"}

        pos_sum = sum(relevant.get(e, 0.0) for e in positive)
        neg_sum = sum(relevant.get(e, 0.0) for e in negative)
        net_valence = pos_sum - neg_sum

        return {
            "active_identity_emotions": relevant,
            "positive_sum": round(pos_sum, 3),
            "negative_sum": round(neg_sum, 3),
            "net_valence": round(net_valence, 3),
            "disruption_risk": neg_sum > 1.0,
        }

    # ------------------------------------------------------------------
    # Pending update analysis
    # ------------------------------------------------------------------

    def _analyse_pending_updates(
        self,
        pending: List[Dict[str, Any]],
        cores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyse pending core memory updates for consistency."""
        pending_count = len([
            p for p in pending if p.get("status") == "pending"
        ])
        approved_count = len([
            p for p in pending if p.get("status") == "approved"
        ])
        rejected_count = len([
            p for p in pending if p.get("status") == "rejected"
        ])

        return {
            "total": len(pending),
            "pending": pending_count,
            "approved": approved_count,
            "rejected": rejected_count,
            "high_confidence_pending": [
                {
                    "update_id": p.get("update_id", ""),
                    "target": p.get("target_memory_id", ""),
                    "confidence": p.get("confidence", 0.0),
                }
                for p in pending
                if p.get("status") == "pending"
                   and p.get("confidence", 0.0) > 0.7
            ],
        }

    # ------------------------------------------------------------------
    # Theme extraction
    # ------------------------------------------------------------------

    def _extract_identity_themes(
        self,
        cores: List[Dict[str, Any]],
        conclusions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract dominant identity themes from core memories + conclusions.

        Groups by memory_type / conclusion_type and counts tag frequencies.
        """
        type_counts: Dict[str, int] = defaultdict(int)
        tag_counts: Dict[str, int] = defaultdict(int)

        for c in cores:
            mtype = c.get("memory_type", "unknown")
            type_counts[f"core:{mtype}"] += 1
            for tag in c.get("tags", []):
                tag_counts[tag] += 1

        for c in conclusions:
            ctype = c.get("conclusion_type", "unknown")
            type_counts[f"conclusion:{ctype}"] += 1
            for tag in c.get("tags", []):
                tag_counts[tag] += 1

        # Top themes by tag frequency
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        themes: List[Dict[str, Any]] = []
        for tag, count in top_tags:
            themes.append({
                "theme": tag,
                "frequency": count,
            })

        return themes

    # ------------------------------------------------------------------
    # Conclusion update recommendations
    # ------------------------------------------------------------------

    def _recommend_conclusion_updates(
        self,
        conclusions: List[Dict[str, Any]],
        fragile: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        self_relevance: float,
    ) -> List[Dict[str, Any]]:
        """Recommend updates to identity conclusions.

        Fragile conclusions with high self-relevance → recommend
        reinforcement seeking or revision.
        Contradicted conclusions → recommend resolution.
        """
        updates: List[Dict[str, Any]] = []

        # Fragile conclusions needing attention
        for f in fragile:
            priority = (1.0 - f.get("confidence", 0.5)) * self_relevance
            updates.append({
                "conclusion_id": f["conclusion_id"],
                "action": "seek_reinforcement",
                "reason": f["reason"],
                "priority": round(priority, 3),
            })

        # Conclusions involved in contradictions
        contradicted_ids: set = set()
        for c in contradictions:
            contradicted_ids.add(c.get("entry_a", ""))
            contradicted_ids.add(c.get("entry_b", ""))

        for conc in conclusions:
            cid = conc.get("conclusion_id", "")
            if cid in contradicted_ids:
                updates.append({
                    "conclusion_id": cid,
                    "action": "resolve_contradiction",
                    "reason": "involved_in_identity_contradiction",
                    "priority": round(0.8 * self_relevance, 3),
                })

        updates.sort(key=lambda x: x["priority"], reverse=True)
        return updates

    # ------------------------------------------------------------------
    # Coherence scoring
    # ------------------------------------------------------------------

    def _compute_coherence_score(
        self,
        *,
        cores: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        fragile_conclusions: List[Dict[str, Any]],
        alignment: Dict[str, Any],
        identity_emotions: Dict[str, Any],
        stability_tolerance: float,
        threat_bias: float,
    ) -> float:
        """Compute composite identity coherence score [0, 1].

        Components:
          - Contradiction penalty: each contradiction reduces score
          - Fragile conclusion penalty: many unstable conclusions reduce score
          - Alignment ratio: high alignment → high score
          - Emotion valence: positive identity emotions → boost
          - Stability tolerance (5-HT): tolerates more issues
        """
        score = 1.0

        # Contradiction penalty (0.1 per contradiction, max 0.5)
        contradiction_penalty = min(0.5, len(contradictions) * 0.1)
        # High severity contradictions penalise more
        high_severity = sum(
            1 for c in contradictions if c.get("severity") == "high"
        )
        contradiction_penalty += min(0.2, high_severity * 0.1)
        score -= contradiction_penalty

        # Fragile conclusion penalty
        total_conclusions = max(len(fragile_conclusions), 1)
        # Normalised: what fraction of conclusions are fragile
        # We don't have total conclusions count here, so use raw count
        fragile_penalty = min(0.3, len(fragile_conclusions) * 0.05)
        score -= fragile_penalty

        # Alignment contribution
        alignment_ratio = alignment.get("alignment_ratio", 1.0)
        alignment_issues = len(alignment.get("alignment_issues", []))
        alignment_penalty = min(0.2, alignment_issues * 0.05)
        score -= alignment_penalty
        score += alignment_ratio * 0.1  # small boost for good alignment

        # Emotion modulation
        net_valence = identity_emotions.get("net_valence", 0.0)
        # Positive emotions slightly boost coherence
        emotion_mod = _clamp(net_valence * 0.1, -0.15, 0.1)
        score += emotion_mod

        # Stability tolerance: 5-HT allows more flexibility
        score += stability_tolerance * 0.1

        # If no core memories exist, baseline is fragmented
        if not cores:
            score = min(score, 0.6)

        return _clamp(score)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_name,
            "active": self._active,
            "cluster": self.cluster,
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_name,
            "coherence_score": 1.0,
            "identity_coherence_status": COHERENCE_COHERENT,
            "identity_contradictions": [],
            "fragile_conclusions": [],
            "alignment_analysis": {},
            "identity_emotions": {},
            "pending_update_analysis": {},
            "identity_themes": [],
            "conclusion_updates": [],
            "cores_analysed": 0,
            "conclusions_analysed": 0,
            "journals_analysed": 0,
            "nt_modulation": {},
        }
