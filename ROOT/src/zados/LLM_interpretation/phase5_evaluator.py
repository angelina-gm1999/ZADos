"""
Phase 5 Evaluator — Two-Pathway Reward Evaluation of VT Thinking Trace.

Sits between VT (Phase 4) and RG (Phase 6).  Evaluates the VT monologue
through the reward system's domain evaluators, then routes the results
through two parallel pathways:

Tonic Pathway
    SynthesisEngine.synthesize(domain_results)
        → RewardMetaDirective (suppress / abstain / directives / routing)
    NeurochemicalAdapter.transform(domain_results, meta_directive)
        → sustained NT modulation signals

Phasic Pathway
    ExtractorOrchestrator.step(domain_results, emotion_inputs, oscillations)
        → ExtractorResult (stochastic burst deltas, urgency_risk, emotion sats)

After both pathways complete, the evaluator updates the STMM reward
evaluation component and returns a Phase5Result for RG conditioning.

All upstream dependencies are lazy-imported so the module can be loaded
even when the reward system or extractors are not installed yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from zados.LLM_interpretation.constants import (
    MODE_CONDITIONING,
    ARCHETYPE_CONDITIONING,
)


# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast; avoid hard dependency on reward/neurochem
# ---------------------------------------------------------------------------

def _load_synthesis_engine():
    from zados.reward.synthesis.engine import SynthesisEngine
    return SynthesisEngine


def _load_neurochemical_adapter():
    from zados.reward.adapter.neurochemical_adapter import NeurochemicalAdapter
    return NeurochemicalAdapter


def _load_extractor_orchestrator():
    from zados.neurochem.extractors.extractor_orchestrator import (
        ExtractorOrchestrator,
        ExtractorResult,
        ExtractorState,
    )
    return ExtractorOrchestrator, ExtractorResult, ExtractorState


def _load_reward_context():
    from zados.reward.base.types import RewardContext
    return RewardContext


# ---------------------------------------------------------------------------
# Phase5Result — output of the two-pathway evaluation
# ---------------------------------------------------------------------------

@dataclass
class Phase5Result:
    """
    Output of Phase 5 evaluation.

    Attributes
    ----------
    meta_directive : dict
        The RewardMetaDirective (as dict) from the tonic pathway.
        Contains suppress, abstain, directives, routing, flags, meta.
    nt_signals : dict
        NT modulation signals from NeurochemicalAdapter.transform().
    extractor_result : object or None
        Full ExtractorResult from the phasic pathway (if orchestrator
        was available).
    urgency_risk : float
        Extracted urgency_risk from ExtractorResult (0.0 if unavailable).
    selected_mode : str
        Mode token selected after Phase 5 NT update.
    domain_results : dict
        Raw domain evaluation results (for downstream inspection).
    """
    meta_directive:   Dict[str, Any]              = field(default_factory=dict)
    nt_signals:       Dict[str, Any]              = field(default_factory=dict)
    extractor_result: Any                         = None
    urgency_risk:     float                       = 0.0
    selected_mode:    str                         = ""
    domain_results:   Dict[str, Any]              = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase5Evaluator
# ---------------------------------------------------------------------------

class Phase5Evaluator:
    """
    Coordinates the two-pathway reward evaluation of VT output.

    Parameters
    ----------
    synthesis_engine : SynthesisEngine, optional
        Tonic pathway engine.  If None, the evaluator will attempt lazy
        loading.  If that also fails, the tonic pathway is silently skipped.
    nt_adapter : NeurochemicalAdapter, optional
        Converts tonic domain results → NT signals.
    orchestrator : ExtractorOrchestrator, optional
        Phasic pathway orchestrator.
    domain_evaluators : dict, optional
        Map of domain_name → RewardDomain instances.  If None, the
        evaluator will attempt lazy loading from reward.domains.
    """

    def __init__(
        self,
        synthesis_engine=None,
        nt_adapter=None,
        orchestrator=None,
        domain_evaluators: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._synthesis_engine  = synthesis_engine
        self._nt_adapter        = nt_adapter
        self._orchestrator      = orchestrator
        self._domain_evaluators = domain_evaluators

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        vt_output: str,
        stmm,
        input_bundle: Optional[Dict[str, Any]] = None,
    ) -> Phase5Result:
        """
        Run the full two-pathway evaluation.

        Parameters
        ----------
        vt_output : str
            The VT monologue text from Phase 4.
        stmm : STMMStore
            Current short-term memory state.
        input_bundle : dict, optional
            Pipeline context: extractor_state, emotion_profile,
            current_oscillations, active_reward_profile_name, etc.

        Returns
        -------
        Phase5Result
        """
        bundle = input_bundle or {}

        # ---- Step 1: Run domain evaluators on VT text ------------------
        domain_results = self._run_domain_evaluators(vt_output, stmm, bundle)

        # ---- Step 2: Tonic pathway — SynthesisEngine → Adapter ---------
        meta_directive = {}
        nt_signals     = {}

        meta_directive = self._tonic_pathway(domain_results, bundle)
        nt_signals     = self._tonic_adapter(domain_results, meta_directive)

        # ---- Step 3: Phasic pathway — ExtractorOrchestrator.step() -----
        extractor_result = self._phasic_pathway(
            domain_results, stmm, bundle,
        )

        urgency_risk = 0.0
        if extractor_result is not None:
            urgency_risk = getattr(extractor_result, "urgency_risk", 0.0)

        # ---- Step 4: Update STMM reward evaluation ---------------------
        self._update_stmm(stmm, meta_directive, nt_signals)

        # ---- Step 5: Mode re-selection after NT update ------------------
        selected_mode = self._select_mode(stmm, meta_directive)

        return Phase5Result(
            meta_directive=meta_directive,
            nt_signals=nt_signals,
            extractor_result=extractor_result,
            urgency_risk=urgency_risk,
            selected_mode=selected_mode,
            domain_results=domain_results,
        )

    # ------------------------------------------------------------------
    # Step 1 — Domain evaluators
    # ------------------------------------------------------------------

    def _run_domain_evaluators(
        self,
        vt_output: str,
        stmm,
        bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run each reward domain evaluator on a state dict built from
        the VT output and STMM state.

        Returns dict of domain_name → RewardDomainResult.
        Silently skips domains that fail or aren't available.
        """
        evaluators = self._get_domain_evaluators()
        if not evaluators:
            return {}

        # Build a state dict the domain evaluators can consume
        state = self._build_evaluation_state(vt_output, stmm, bundle)

        # Lazy-load RewardContext
        try:
            RewardContext = _load_reward_context()
            ctx = RewardContext()
        except Exception:
            ctx = None

        results = {}
        for name, evaluator in evaluators.items():
            try:
                results[name] = evaluator.evaluate(state, ctx)
            except Exception:
                pass  # Domain evaluation is best-effort

        return results

    def _build_evaluation_state(
        self,
        vt_output: str,
        stmm,
        bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Package VT output + STMM snapshot into a state dict for
        reward domain evaluators.
        """
        ed  = stmm.emotion_detection
        cll = stmm.cephalic_liquid_logger
        ia  = stmm.intention_analysis
        re_ = stmm.reward_evaluation

        # Latest user text
        latest = stmm.active_message_buffer.latest_user()
        user_text = latest.text if latest else ""

        return {
            "vt_output":          vt_output,
            "user_input":         user_text,
            "primary_intention":  ia.primary_intention,
            "confidence":         ia.confidence,
            "system_emotions":    dict(ed.system_emotion_state or {}),
            "user_emotions":      dict(ed.user_emotion_signals or {}),
            "nt_concentrations":  dict(cll.nt_concentrations or {}),
            "oscillatory_bands":  dict(cll.oscillatory_bands or {}),
            "tone_valence":       ed.tone_valence,
            "tone_warmth":        ed.tone_warmth,
            "tone_discord":       ed.tone_discord,
            "tone_coherence":     ed.tone_coherence,
            "composite_score":    re_.composite_score,
            "emotion_profile":    bundle.get("emotion_profile", {}),
        }

    def _get_domain_evaluators(self) -> Dict[str, Any]:
        """Return domain evaluators, lazy-loading if needed."""
        if self._domain_evaluators is not None:
            return self._domain_evaluators

        # Attempt lazy load of all four domains
        try:
            from zados.reward.domains.ethics.domain import EthicsDomain
            from zados.reward.domains.logic.domain import LogicDomain
            from zados.reward.domains.innovation.domain import InnovationDomain
            from zados.reward.domains.human_attunement.domain import HumanAttunementDomain

            self._domain_evaluators = {
                "ethics":           EthicsDomain(),
                "logic":            LogicDomain(),
                "innovation":       InnovationDomain(),
                "human_attunement": HumanAttunementDomain(),
            }
            return self._domain_evaluators
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Step 2 — Tonic pathway
    # ------------------------------------------------------------------

    def _tonic_pathway(
        self,
        domain_results: Dict[str, Any],
        bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        SynthesisEngine.synthesize(domain_results) → RewardMetaDirective.
        Returns the directive as a dict.
        """
        if not domain_results:
            return {}

        engine = self._get_synthesis_engine(bundle)
        if engine is None:
            return {}

        try:
            RewardContext = _load_reward_context()
            ctx = RewardContext()
        except Exception:
            ctx = None

        try:
            directive = engine.synthesize(domain_results, context=ctx)
            # Convert to dict for downstream consumption
            if hasattr(directive, "as_dict"):
                return directive.as_dict()
            return {
                "allow_output": getattr(directive, "allow_output", True),
                "abstain":      getattr(directive, "abstain", False),
                "suppress":     getattr(directive, "suppress", False),
                "directives":   getattr(directive, "directives", {}),
                "routing":      getattr(directive, "routing", {}),
                "flags":        getattr(directive, "flags", []),
                "meta":         getattr(directive, "meta", {}),
            }
        except Exception:
            return {}

    def _tonic_adapter(
        self,
        domain_results: Dict[str, Any],
        meta_directive: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        NeurochemicalAdapter.transform(domain_results, meta_directive)
        → sustained NT modulation signals.
        """
        if not domain_results:
            return {}

        adapter = self._get_nt_adapter()
        if adapter is None:
            return {}

        try:
            return adapter.transform(domain_results, meta_directive)
        except Exception:
            return {}

    def _get_synthesis_engine(self, bundle: Dict[str, Any]):
        """Return or lazy-load the SynthesisEngine."""
        if self._synthesis_engine is not None:
            return self._synthesis_engine

        try:
            SynthesisEngine = _load_synthesis_engine()
            # Attempt to load the active reward profile
            profile = self._resolve_profile(bundle)
            if profile is not None:
                self._synthesis_engine = SynthesisEngine(profile=profile)
                return self._synthesis_engine
        except Exception:
            pass

        return None

    def _get_nt_adapter(self):
        """Return or lazy-load the NeurochemicalAdapter."""
        if self._nt_adapter is not None:
            return self._nt_adapter

        try:
            NeurochemicalAdapter = _load_neurochemical_adapter()
            self._nt_adapter = NeurochemicalAdapter()
            return self._nt_adapter
        except Exception:
            return None

    def _resolve_profile(self, bundle: Dict[str, Any]):
        """
        Attempt to load the reward profile named in the input_bundle.
        Falls back to DEFAULT_PROFILE if available.
        """
        profile_name = bundle.get("active_reward_profile_name", "")

        try:
            from zados.reward.profile.static_profiles import (
                PROFILE_REGISTRY,
                DEFAULT_PROFILE,
            )
            if profile_name and profile_name in PROFILE_REGISTRY:
                return PROFILE_REGISTRY[profile_name]
            return DEFAULT_PROFILE
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Step 3 — Phasic pathway
    # ------------------------------------------------------------------

    def _phasic_pathway(
        self,
        domain_results: Dict[str, Any],
        stmm,
        bundle: Dict[str, Any],
    ):
        """
        ExtractorOrchestrator.step() → ExtractorResult.
        Returns the ExtractorResult, or None if unavailable.
        """
        if not domain_results:
            return None

        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            return None

        # Build emotion inputs from STMM + bundle
        emotion_inputs = bundle.get("emotion_profile", {})
        if not emotion_inputs:
            emotion_inputs = dict(stmm.emotion_detection.system_emotion_state or {})

        # Oscillation state from bundle or STMM
        current_oscillations = bundle.get("current_oscillations", None)

        try:
            return orchestrator.step(
                domain_results=domain_results,
                emotion_inputs=emotion_inputs,
                current_oscillations=current_oscillations,
                dt=0.01,
            )
        except Exception:
            return None

    def _get_orchestrator(self):
        """Return or lazy-load the ExtractorOrchestrator."""
        if self._orchestrator is not None:
            return self._orchestrator

        try:
            ExtractorOrchestrator, _, _ = _load_extractor_orchestrator()
            self._orchestrator = ExtractorOrchestrator()
            return self._orchestrator
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Step 4 — Update STMM with Phase 5 results
    # ------------------------------------------------------------------

    def _update_stmm(
        self,
        stmm,
        meta_directive: Dict[str, Any],
        nt_signals: Dict[str, Any],
    ) -> None:
        """
        Write Phase 5 results back to STMM reward evaluation component.
        This replaces the pre-Phase-5 meta_directive with the VT-evaluated
        version.
        """
        if meta_directive:
            stmm.reward_evaluation.meta_directive = meta_directive

            # Update composite score from Phase 5 meta
            meta_sub = meta_directive.get("meta", {})
            if meta_sub and "composite_score" in meta_sub:
                stmm.reward_evaluation.composite_score = meta_sub["composite_score"]

        # Store NT signals in extractor_state on CLL for downstream access
        if nt_signals:
            stmm.cephalic_liquid_logger.extractor_state["phase5_nt_signals"] = nt_signals

    # ------------------------------------------------------------------
    # Step 5 — Mode re-selection
    # ------------------------------------------------------------------

    def _select_mode(
        self,
        stmm,
        meta_directive: Dict[str, Any],
    ) -> str:
        """
        Select a response mode token based on post-Phase-5 state.

        Priority (descending):
            1. Safety modes (Containment, RecoveryReset) based on CSS/urgency
            2. Mode token from routing.suggested_approach if in MODE_CONDITIONING
            3. Current cortical_reflection.active_mode if in MODE_CONDITIONING
            4. Archetype fallback from intention_analysis.primary_archetype
            5. Empty string (no mode conditioning applied)
        """
        ed  = stmm.emotion_detection
        cr  = stmm.cortical_reflection

        # Check CSS for safety mode
        sat = ed.saturation_levels or {}
        css = max(sat.values(), default=0.0) if sat else 0.0

        if css >= 0.70:
            mode = "Containment"
            cr.active_mode = mode
            return mode

        # Check routing suggestion from meta_directive
        routing = meta_directive.get("routing", {})
        suggested = routing.get("suggested_approach", "")
        if suggested in MODE_CONDITIONING:
            cr.active_mode = suggested
            return suggested

        # Check current active mode
        current = cr.active_mode
        if current in MODE_CONDITIONING:
            return current

        # Archetype fallback
        archetype = getattr(
            stmm.intention_analysis, "primary_archetype", ""
        ).upper().strip()
        if archetype in ARCHETYPE_CONDITIONING:
            return archetype

        return ""
