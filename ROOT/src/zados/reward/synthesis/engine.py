from __future__ import annotations

from typing import Dict, Any, Optional

from zados.reward.base.types import (
    RewardDomainResult,
    RewardMetaDirective,
    RewardContext,
)
from zados.reward.profile.base import RewardProfile
from zados.reward.synthesis.directives import (
    classify_tier,
    tier_label,
    compute_weighted_composite,
    compute_per_domain_weighted_scores,
    compute_suppression,
    compute_abstention,
    compute_response_directives,
    compute_routing,
    escalate_domain_flags,
    apply_cross_domain_interactions,
)


class SynthesisEngine:
    """
    Combines 4 reward domain results into a RewardMetaDirective.

    Stateless orchestrator -- all computation delegated to pure functions
    in ``zados.reward.synthesis.directives``.  Configuration comes from a
    :class:`RewardProfile`.

    3-Step Synthesis Flow (from masterdoc)
    --------------------------------------
    1. Interpret subdomains  ->  per-domain tier classification
    2. Apply reward weighting  ->  weighted composite score R(t)
    3. Composite synthesis  ->  final RewardMetaDirective

    Usage
    -----
    >>> from zados.reward.profile.static_profiles import REFLECTIVE_PROFILE
    >>> engine = SynthesisEngine(profile=REFLECTIVE_PROFILE)
    >>> directive = engine.synthesize(domain_results)
    >>> adapter.transform(domain_results, meta_directive=directive)
    """

    def __init__(self, profile: RewardProfile) -> None:
        self._profile = profile

    @property
    def profile(self) -> RewardProfile:
        """The active reward profile."""
        return self._profile

    def synthesize(
        self,
        domain_results: Dict[str, RewardDomainResult],
        context: Optional[RewardContext] = None,
        sleep_metrics: Optional[Dict[str, float]] = None,
    ) -> RewardMetaDirective:
        """
        Synthesize domain results into a RewardMetaDirective.

        Parameters
        ----------
        domain_results : dict
            Map of domain_name -> RewardDomainResult.
            Expected keys: ``"ethics"``, ``"logic"``, ``"innovation"``,
            ``"human_attunement"``.  Missing domains are handled gracefully.
        context : RewardContext, optional
            Evaluation context (reward_profile, timestamp).
        sleep_metrics : dict, optional
            Sleep-state metrics from the neurochem layer.  Keys:
            ``"dream_permissiveness"``, ``"consolidation_depth"``,
            ``"narrative_plasticity"``.  When present, modulates
            suppression and abstention thresholds.

        Returns
        -------
        RewardMetaDirective
            Frozen dataclass with allow_output, abstain, suppress,
            directives, routing, flags, and meta.
        """
        ctx = context or RewardContext()
        weights = self._profile.domain_weights

        # Step 1: Classify each domain into a 4-tier influence level
        per_domain_tiers = {
            name: classify_tier(result.general_score)
            for name, result in domain_results.items()
        }

        # Step 2: Compute weighted composite score R(t)
        composite = compute_weighted_composite(domain_results, weights)
        per_domain_weighted = compute_per_domain_weighted_scores(
            domain_results, weights,
        )

        # Step 3a: Flag escalation
        meta_flags, domain_flag_lists = escalate_domain_flags(domain_results)

        # Step 3b: Suppression decision
        suppression_bias = self._profile.suppression_bias
        abstention_bias = self._profile.abstention_bias

        # Sleep metric modulation: during consolidation or dreaming,
        # relax suppression/abstention thresholds
        sm = sleep_metrics or {}
        consolidation = sm.get("consolidation_depth", 0.0)
        dream_perm = sm.get("dream_permissiveness", 0.0)

        if consolidation > 0.5:
            # During deep consolidation, reduce suppression sensitivity
            suppression_bias = suppression_bias * (1.0 - 0.5 * consolidation)
        if dream_perm > 0.5:
            # During dreaming, reduce abstention (allow creative output)
            abstention_bias = abstention_bias * (1.0 - 0.5 * dream_perm)

        suppress = compute_suppression(
            composite,
            suppression_bias,
            domain_flag_lists,
        )

        # Step 3c: Abstention decision
        abstain = compute_abstention(
            domain_results,
            self._profile.threshold_tolerances,
            abstention_bias,
        )

        # Step 3d: allow_output is False if either suppress or abstain
        allow_output = not (suppress or abstain)

        # Step 3e: Compute response-shaping directives
        directives = compute_response_directives(
            domain_results, weights, per_domain_tiers,
        )

        # Step 3f: Apply cross-domain interaction effects
        directives = apply_cross_domain_interactions(
            directives, domain_results, weights,
        )

        # Step 3g: Compute routing hints
        routing = compute_routing(per_domain_tiers, weights, composite)

        # Step 3h: Assemble metadata
        meta: Dict[str, Any] = {
            "profile_name": self._profile.name,
            "composite_score": composite,
            "composite_tier": classify_tier(composite),
            "composite_tier_label": tier_label(classify_tier(composite)),
            "per_domain_weighted_scores": per_domain_weighted,
            "per_domain_tiers": per_domain_tiers,
            "per_domain_tier_labels": {
                name: tier_label(tier)
                for name, tier in per_domain_tiers.items()
            },
            "context_reward_profile": ctx.reward_profile,
        }

        if sm:
            meta["sleep_metrics"] = dict(sm)

        return RewardMetaDirective(
            allow_output=allow_output,
            abstain=abstain,
            suppress=suppress,
            directives=directives,
            routing=routing,
            flags=meta_flags,
            meta=meta,
        )
