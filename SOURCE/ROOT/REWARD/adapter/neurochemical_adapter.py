from __future__ import annotations

from typing import Dict, Any, Optional

from zados.reward.base.types import RewardDomainResult, RewardMetaDirective
from zados.reward.adapter.mapping import (
    map_innovation_to_dopamine,
    map_logic_to_norepinephrine,
    map_attunement_to_oxytocin,
    map_ethics_to_constraint_awareness,
    map_flags_to_stress_response,
    compute_motivation_modulation,
)


class NeurochemicalAdapter:
    """
    Adapter layer between reward system and neurochemical simulator.
    
    Transforms reward domain evaluations into neurochemical modulation signals.
    
    Stateless transformation layer - no internal state, no side effects.
    
    Usage
    -----
    >>> adapter = NeurochemicalAdapter()
    >>> domain_results = {
    ...     "innovation": innovation_domain.evaluate(state, ctx),
    ...     "logic": logic_domain.evaluate(state, ctx),
    ...     "human_attunement": attunement_domain.evaluate(state, ctx),
    ...     "ethics": ethics_domain.evaluate(state, ctx),
    ... }
    >>> nt_signals = adapter.transform(domain_results)
    """
    
    def transform(
        self,
        domain_results: Dict[str, Any],
        meta_directive: Optional[RewardMetaDirective] = None,
        sleep_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Transform reward domain results into neurochemical modulation signals.
        
        Parameters
        ----------
        domain_results : dict
            Map of domain_name → RewardDomainResult
            Expected keys: "innovation", "logic", "human_attunement", "ethics"
        meta_directive : RewardMetaDirective, optional
            Synthesis output (allow/suppress/abstain)
            
        Returns
        -------
        dict
            Neurochemical modulation signals structured as:
            {
                "DA": {"novelty": float, "rpe": float, "effort": float},
                "NE": {"precision": float, "uncertainty": float},
                "OXT": {"empathy": float, "social_engagement": float},
                "cortisol": {"level": float},
                "CRH": {"level": float},
                "motivation_modulation": float,
            }
        """
        # Extract domain results
        innovation = self._extract_domain(domain_results, "innovation")
        logic = self._extract_domain(domain_results, "logic")
        attunement = self._extract_domain(domain_results, "human_attunement")
        ethics = self._extract_domain(domain_results, "ethics")
        
        # Aggregate all flags across domains
        all_flags = self._aggregate_flags(domain_results)
        
        # Map domains to NT signals
        da_signals = map_innovation_to_dopamine(innovation)
        ne_signals = map_logic_to_norepinephrine(logic)
        oxt_signals = map_attunement_to_oxytocin(attunement)
        constraint_signals = map_ethics_to_constraint_awareness(ethics)
        stress_signals = map_flags_to_stress_response(all_flags)
        
        # Compute motivation modulation
        meta_dict = meta_directive.as_dict() if meta_directive else None
        motivation = compute_motivation_modulation(meta_dict, da_signals)
        
        # Assemble final signal structure
        signals = {
            "DA": {
                "novelty": da_signals["novelty"],
                "rpe": da_signals["rpe"],
                "effort": 0.0,  # No effort signal from current domains
            },
            "NE": {
                "precision": ne_signals["precision"],
                "uncertainty": ne_signals["uncertainty"],
            },
            "OXT": {
                "empathy": oxt_signals["empathy"],
                "social_engagement": oxt_signals["social_engagement"],
            },
            "cortisol": {
                "level": stress_signals["cortisol"],
            },
            "CRH": {
                "level": stress_signals["crh"],
            },
            "GABA": {
                "inhibition": constraint_signals["boundary_proximity"],
            },
            "motivation_modulation": motivation,
            "meta": {
                "risk_awareness": constraint_signals["risk_awareness"],
                "tonic_da_bias": da_signals["tonic_bias"],
            },
        }

        # Sleep metric → NT signal injection
        sm = sleep_metrics or {}
        if sm:
            dream_perm = sm.get("dream_permissiveness", 0.0)
            consolidation = sm.get("consolidation_depth", 0.0)
            narrative = sm.get("narrative_plasticity", 0.0)

            if dream_perm > 0.0:
                signals["DA"]["dream_release_boost"] = dream_perm * 0.3
                signals.setdefault("CB1", {})["dream_baseline_boost"] = dream_perm * 0.2
            if consolidation > 0.0:
                signals["GABA"]["consolidation_baseline_boost"] = consolidation * 0.25
                signals.setdefault("GLU", {})["consolidation_nmda_boost"] = consolidation * 0.2
            if narrative > 0.0:
                signals["DA"]["narrative_d3_boost"] = narrative * 0.2
            signals["meta"]["sleep_metrics"] = dict(sm)

        return signals
    
    def _extract_domain(
        self,
        domain_results: Dict[str, Any],
        domain_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract and convert domain result to dict."""
        result = domain_results.get(domain_name)
        
        if result is None:
            return None
        
        # If it's a RewardDomainResult object, convert to dict
        if hasattr(result, "subscores"):
            return {
                "domain": result.domain,
                "general_score": result.general_score,
                "subscores": result.subscores,
                "flags": result.flags,
                "meta": result.meta,
            }
        
        # Already a dict
        return result
    
    def _aggregate_flags(
        self,
        domain_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate flags from all domains."""
        all_flags = {}
        
        for domain_name, result in domain_results.items():
            if result is None:
                continue
            
            flags = result.flags if hasattr(result, "flags") else result.get("flags", {})
            
            for flag_name, flag_obj in flags.items():
                # Prefix with domain to avoid collisions
                key = f"{domain_name}_{flag_name}"
                all_flags[key] = flag_obj
        
        return all_flags