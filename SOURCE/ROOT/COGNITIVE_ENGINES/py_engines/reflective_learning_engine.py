"""
ZA-DOS v0.6 — E31: Reflective Learning Engine (Appendix §4.1).

Meta-learning engine that analyses learning log history to detect:
  1. Recurring learning failures (same error type across sessions)
  2. Learning mode effectiveness (confirmation/contradiction ratios per mode)
  3. Subject proficiency trends (improving vs stagnating domains)
  4. Learning style preferences (which modes produce deepest encoding)

NT coupling:
  - DA modulates pattern salience: high DA → focus on high-reward patterns
  - 5-HT modulates abstraction level: high 5-HT → broader meta-patterns
  - ACh modulates precision: high ACh → finer-grained failure detection
  - NE modulates urgency: high NE → prioritise recurring failures

Input:
  learning_entries  — list of LearningLogEntry dicts
  session_history   — optional list of session summaries
  identity_context  — optional dict of core memory themes

Output:
  learning_patterns       — detected meta-patterns across entries
  recurring_failures      — repeated error types / misconceptions
  mode_effectiveness      — per-mode stats (confirmations, contradictions, etc.)
  style_preferences       — ranked modes by encoding depth
  subject_proficiencies   — per-subject trend (improving/stagnating/declining)
  recommendations         — actionable suggestions for learning strategy
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from zados.cognitive_engines.constants import _clamp

log = logging.getLogger(__name__)


class ReflectiveLearningEngine:
    """Engine 31 — Reflective Learning: meta-learning pattern analysis."""

    engine_id = 31
    engine_name = "reflective_learning_engine"
    cluster = "metacognition"

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------
    _RECURRING_FAILURE_MIN_COUNT = 2      # min times same error type appears
    _STAGNATION_THRESHOLD = 0.3           # contradiction ratio above this = stagnating
    _PROFICIENCY_WINDOW = 10              # entries per subject to compute trend
    _LOW_CONFIRMATION_RATIO = 0.4         # below this, mode is underperforming

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
            Canonical lowercase keys (da, 5ht, ne, ach, etc.).
        """
        self._nt_state = dict(nt_state)

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(
        self,
        *,
        learning_entries: Optional[List[Dict[str, Any]]] = None,
        session_history: Optional[List[Dict[str, Any]]] = None,
        identity_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Analyse learning logs for meta-learning patterns.

        Parameters
        ----------
        learning_entries : list of dict
            Each dict represents a LearningLogEntry (or its serialised form).
            Expected keys: mode, subject, confirmations, contradictions,
            extensions, novel_entries, patterns_detected, reward_scores, etc.
        session_history : list of dict, optional
            Session-level summaries for cross-session trend analysis.
        identity_context : dict, optional
            Core memory themes / identity conclusions for relevance weighting.

        Returns
        -------
        Dict[str, Any]
        """
        entries = learning_entries or []
        if not entries:
            log.debug("E31: No learning entries to analyse.")
            return self._empty_result()

        # NT modulation factors
        da = _clamp(self._nt_state.get("da", 0.5))
        sht = _clamp(self._nt_state.get("5ht", 0.5))
        ach = _clamp(self._nt_state.get("ach", 0.5))
        ne = _clamp(self._nt_state.get("ne", 0.5))

        # Salience threshold: high DA focuses on high-reward patterns
        salience_threshold = 0.3 + 0.4 * da  # range [0.3, 0.7]

        # Abstraction level: high 5-HT → broader grouping
        abstraction_level = 0.3 + 0.4 * sht  # range [0.3, 0.7]

        # Precision: high ACh → finer error detection
        precision_factor = 0.5 + 0.5 * ach   # range [0.5, 1.0]

        # Urgency: high NE → prioritise failures over neutral patterns
        urgency_weight = 0.3 + 0.7 * ne      # range [0.3, 1.0]

        # ---- Analysis passes ----
        mode_stats = self._compute_mode_effectiveness(entries)
        subject_stats = self._compute_subject_proficiencies(entries)
        recurring = self._detect_recurring_failures(entries, precision_factor)
        patterns = self._detect_meta_patterns(
            entries, mode_stats, subject_stats,
            salience_threshold, abstraction_level,
        )
        style_prefs = self._rank_mode_preferences(mode_stats)
        recommendations = self._generate_recommendations(
            mode_stats, subject_stats, recurring, urgency_weight,
        )

        result = {
            "engine_id": self.engine_name,
            "entries_analysed": len(entries),
            "learning_patterns": patterns,
            "recurring_failures": recurring,
            "mode_effectiveness": mode_stats,
            "style_preferences": style_prefs,
            "subject_proficiencies": subject_stats,
            "recommendations": recommendations,
            "nt_modulation": {
                "salience_threshold": round(salience_threshold, 3),
                "abstraction_level": round(abstraction_level, 3),
                "precision_factor": round(precision_factor, 3),
                "urgency_weight": round(urgency_weight, 3),
            },
        }

        log.info(
            "E31: Analysed %d entries — %d patterns, %d recurring failures, "
            "%d recommendations.",
            len(entries), len(patterns), len(recurring), len(recommendations),
        )
        return result

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _compute_mode_effectiveness(
        self,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-mode learning statistics.

        Returns dict keyed by mode (M1..M5) with counts:
          turns, confirmations, contradictions, extensions,
          novel_entries, patterns_detected, confirmation_ratio.
        """
        stats: Dict[str, Dict[str, Any]] = {}

        for entry in entries:
            mode = entry.get("mode", "unknown")
            if mode not in stats:
                stats[mode] = {
                    "turns": 0,
                    "confirmations": 0,
                    "contradictions": 0,
                    "extensions": 0,
                    "novel_entries": 0,
                    "patterns_detected": 0,
                }
            s = stats[mode]
            s["turns"] += 1
            s["confirmations"] += _safe_count(entry.get("confirmations"))
            s["contradictions"] += _safe_count(entry.get("contradictions"))
            s["extensions"] += _safe_count(entry.get("extensions"))
            s["novel_entries"] += _safe_count(entry.get("novel_entries"))
            s["patterns_detected"] += _safe_count(entry.get("patterns_detected"))

        # Compute ratios
        for mode, s in stats.items():
            total = s["confirmations"] + s["contradictions"]
            s["confirmation_ratio"] = (
                round(s["confirmations"] / total, 3) if total > 0 else 0.0
            )
            s["contradiction_ratio"] = (
                round(s["contradictions"] / total, 3) if total > 0 else 0.0
            )

        return stats

    def _compute_subject_proficiencies(
        self,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-subject proficiency trends.

        Groups entries by subject, computes confirmation ratio for
        first half vs second half to detect improvement/stagnation.
        """
        by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            subj = entry.get("subject", "unknown")
            by_subject[subj].append(entry)

        result: Dict[str, Dict[str, Any]] = {}
        for subj, subj_entries in by_subject.items():
            total_conf = sum(_safe_count(e.get("confirmations")) for e in subj_entries)
            total_contra = sum(_safe_count(e.get("contradictions")) for e in subj_entries)
            total = total_conf + total_contra

            # Trend: compare first half vs second half
            n = len(subj_entries)
            mid = max(1, n // 2)
            first_half = subj_entries[:mid]
            second_half = subj_entries[mid:]

            first_ratio = _confirmation_ratio(first_half)
            second_ratio = _confirmation_ratio(second_half)
            delta = second_ratio - first_ratio

            if delta > 0.1:
                trend = "improving"
            elif delta < -0.1:
                trend = "declining"
            else:
                trend = "stable"

            # Check for stagnation: high contradiction ratio overall
            overall_ratio = total_contra / total if total > 0 else 0.0
            if overall_ratio > self._STAGNATION_THRESHOLD and trend != "improving":
                trend = "stagnating"

            result[subj] = {
                "entries": n,
                "confirmations": total_conf,
                "contradictions": total_contra,
                "confirmation_ratio": round(total_conf / total, 3) if total > 0 else 0.0,
                "trend": trend,
                "trend_delta": round(delta, 3),
            }

        return result

    def _detect_recurring_failures(
        self,
        entries: List[Dict[str, Any]],
        precision_factor: float,
    ) -> List[Dict[str, Any]]:
        """Detect repeated error types across entries.

        A 'failure' is an entry with contradictions > confirmations, or
        with specific contradiction types that recur.

        Higher ACh (precision_factor) lowers the min count threshold.
        """
        min_count = max(
            1,
            int(self._RECURRING_FAILURE_MIN_COUNT / precision_factor),
        )

        # Collect contradiction descriptions
        contradiction_types: Counter = Counter()
        contradiction_subjects: Dict[str, List[str]] = defaultdict(list)

        for entry in entries:
            contras = entry.get("contradictions", [])
            if isinstance(contras, int):
                if contras > 0:
                    subj = entry.get("subject", "unknown")
                    key = f"unspecified_contradiction:{subj}"
                    contradiction_types[key] += contras
                    contradiction_subjects[key].append(subj)
            elif isinstance(contras, list):
                for c in contras:
                    desc = c if isinstance(c, str) else str(c)
                    contradiction_types[desc] += 1
                    subj = entry.get("subject", "unknown")
                    contradiction_subjects[desc].append(subj)

        # Filter by min_count
        recurring: List[Dict[str, Any]] = []
        for desc, count in contradiction_types.most_common():
            if count >= min_count:
                subjects = list(set(contradiction_subjects.get(desc, [])))
                recurring.append({
                    "failure_type": desc,
                    "occurrences": count,
                    "subjects": subjects,
                    "severity": "high" if count >= min_count * 2 else "medium",
                })

        return recurring

    def _detect_meta_patterns(
        self,
        entries: List[Dict[str, Any]],
        mode_stats: Dict[str, Dict[str, Any]],
        subject_stats: Dict[str, Dict[str, Any]],
        salience_threshold: float,
        abstraction_level: float,
    ) -> List[Dict[str, Any]]:
        """Detect high-level meta-patterns across learning history.

        Examples:
          - Mode switching patterns (always switches from M1→M3)
          - Subject avoidance (subject with 0 recent entries)
          - Confirmation bias (very high confirmation ratio → not learning)
          - Comfort zone (only using one mode repeatedly)
        """
        patterns: List[Dict[str, Any]] = []

        # Pattern: Comfort zone — one mode dominates (>70% of entries)
        total_turns = sum(s["turns"] for s in mode_stats.values())
        if total_turns > 0:
            for mode, s in mode_stats.items():
                ratio = s["turns"] / total_turns
                if ratio > 0.7 and total_turns >= 5:
                    patterns.append({
                        "pattern_type": "comfort_zone",
                        "description": (
                            f"Mode {mode} used {ratio:.0%} of the time "
                            f"({s['turns']}/{total_turns} turns). Consider "
                            f"diversifying learning approaches."
                        ),
                        "mode": mode,
                        "ratio": round(ratio, 3),
                        "salience": round(ratio, 3),
                    })

        # Pattern: Confirmation bias — very high confirmation ratio
        for mode, s in mode_stats.items():
            if s["turns"] >= 3 and s["confirmation_ratio"] > 0.95:
                patterns.append({
                    "pattern_type": "confirmation_bias_risk",
                    "description": (
                        f"Mode {mode} shows {s['confirmation_ratio']:.0%} "
                        f"confirmation rate — may indicate insufficient "
                        f"challenge depth."
                    ),
                    "mode": mode,
                    "salience": round(1.0 - s["confirmation_ratio"], 3),
                })

        # Pattern: Underperforming mode
        for mode, s in mode_stats.items():
            if (s["turns"] >= 3
                    and s["confirmation_ratio"] < self._LOW_CONFIRMATION_RATIO):
                patterns.append({
                    "pattern_type": "underperforming_mode",
                    "description": (
                        f"Mode {mode} has low confirmation rate "
                        f"({s['confirmation_ratio']:.0%}). May need "
                        f"different approach or prerequisite knowledge."
                    ),
                    "mode": mode,
                    "salience": round(
                        self._LOW_CONFIRMATION_RATIO - s["confirmation_ratio"],
                        3,
                    ),
                })

        # Pattern: Stagnating subjects (from subject proficiencies)
        for subj, sp in subject_stats.items():
            if sp["trend"] == "stagnating":
                patterns.append({
                    "pattern_type": "subject_stagnation",
                    "description": (
                        f"Subject '{subj}' shows stagnation — high "
                        f"contradiction ratio ({sp['confirmation_ratio']:.0%} "
                        f"confirmation) without improvement trend."
                    ),
                    "subject": subj,
                    "salience": round(1.0 - sp["confirmation_ratio"], 3),
                })

        # Filter by salience threshold (DA-modulated)
        filtered = [
            p for p in patterns
            if p.get("salience", 0.0) >= salience_threshold
               or salience_threshold < 0.35  # low DA → accept all
        ]

        # At high abstraction (5-HT), group similar patterns
        if abstraction_level > 0.6:
            # Merge same pattern_type entries
            by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for p in filtered:
                by_type[p["pattern_type"]].append(p)

            merged: List[Dict[str, Any]] = []
            for ptype, group in by_type.items():
                if len(group) > 1:
                    merged.append({
                        "pattern_type": ptype,
                        "description": (
                            f"Multiple instances of {ptype} detected "
                            f"({len(group)} occurrences)."
                        ),
                        "instances": group,
                        "salience": max(p.get("salience", 0.0) for p in group),
                    })
                else:
                    merged.extend(group)
            return merged

        return filtered

    def _rank_mode_preferences(
        self,
        mode_stats: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank learning modes by encoding effectiveness.

        Scoring: weighted sum of confirmation_ratio, novel_entries
        per turn, and extensions per turn.
        """
        ranked: List[Dict[str, Any]] = []
        for mode, s in mode_stats.items():
            turns = max(s["turns"], 1)
            score = (
                s["confirmation_ratio"] * 0.4
                + min(s["novel_entries"] / turns, 1.0) * 0.3
                + min(s["extensions"] / turns, 1.0) * 0.2
                + min(s["patterns_detected"] / turns, 1.0) * 0.1
            )
            ranked.append({
                "mode": mode,
                "effectiveness_score": round(score, 3),
                "turns": turns,
                "novel_per_turn": round(s["novel_entries"] / turns, 3),
                "confirmation_ratio": s["confirmation_ratio"],
            })

        ranked.sort(key=lambda x: x["effectiveness_score"], reverse=True)
        return ranked

    def _generate_recommendations(
        self,
        mode_stats: Dict[str, Dict[str, Any]],
        subject_stats: Dict[str, Dict[str, Any]],
        recurring_failures: List[Dict[str, Any]],
        urgency_weight: float,
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from analysis.

        Higher NE (urgency_weight) prioritises failure-related
        recommendations over neutral optimisation suggestions.
        """
        recs: List[Dict[str, Any]] = []

        # Recurring failures → recommend M2 peer review
        for failure in recurring_failures:
            priority = min(1.0, failure["occurrences"] * 0.2) * urgency_weight
            recs.append({
                "recommendation": (
                    f"Recurring failure '{failure['failure_type']}' "
                    f"({failure['occurrences']} times). Consider M2 peer "
                    f"review on subjects: {', '.join(failure['subjects'])}."
                ),
                "action": "suggest_m2_review",
                "priority": round(priority, 3),
                "target_subjects": failure["subjects"],
            })

        # Stagnating subjects → recommend mode switch
        for subj, sp in subject_stats.items():
            if sp["trend"] == "stagnating":
                priority = 0.6 * urgency_weight
                recs.append({
                    "recommendation": (
                        f"Subject '{subj}' is stagnating. Try M3 (Learn "
                        f"Together) for dialectic exploration or M5 "
                        f"(Independent Study) for fresh perspectives."
                    ),
                    "action": "suggest_mode_switch",
                    "priority": round(priority, 3),
                    "target_subject": subj,
                })

        # Sort by priority (highest first)
        recs.sort(key=lambda x: x["priority"], reverse=True)
        return recs

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
            "entries_analysed": 0,
            "learning_patterns": [],
            "recurring_failures": [],
            "mode_effectiveness": {},
            "style_preferences": [],
            "subject_proficiencies": {},
            "recommendations": [],
            "nt_modulation": {},
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _safe_count(value: Any) -> int:
    """Safely extract a count from a value (int, list, or None)."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return 0


def _confirmation_ratio(entries: List[Dict[str, Any]]) -> float:
    """Compute confirmation ratio for a list of entries."""
    total_conf = sum(_safe_count(e.get("confirmations")) for e in entries)
    total_contra = sum(_safe_count(e.get("contradictions")) for e in entries)
    total = total_conf + total_contra
    return total_conf / total if total > 0 else 0.0
