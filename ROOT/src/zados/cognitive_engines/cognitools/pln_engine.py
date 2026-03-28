"""
Engine 10 — PLN Core  (Probabilistic Logic Networks)
=====================================================

Backward-chaining probabilistic inference engine operating over
AtomSpace (E9) atoms.  Each of the 12 inference rules is a **pure
function** that takes input TruthValues and returns an output TruthValue,
properly propagating uncertainty through the derivation.

Key concepts
------------
- Every inference rule has a **confidence factor** < 1.0 — longer chains
  naturally erode confidence, making the system trust direct evidence
  more than long reasoning chains.
- Backward chaining: given a target atom, recursively find premises
  that could prove it, applying rules until the step budget is
  exhausted.
- Neurochemical modulation controls depth (NE), boldness (DA),
  conservatism (5-HT), pruning (GABA), and risk aversion (cortisol).

Spec: ``docs/specs/engine_10_pln_spec.md``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.cognitools.atomspace_engine import (
    AtomSpaceEngine,
    AtomType,
    Atom,
    TruthValue,
)

# =========================================================================
# 1. Rule Types
# =========================================================================

class RuleType(str, Enum):
    DEDUCTION                = "deduction"
    INDUCTION                = "induction"
    ABDUCTION                = "abduction"
    MODUS_PONENS             = "modus_ponens"
    REVISION                 = "revision"
    AND                      = "and_rule"
    OR                       = "or_rule"
    NOT                      = "not_rule"
    SIMILARITY_TO_INHERITANCE = "similarity_to_inheritance"
    INHERITANCE_TO_SIMILARITY = "inheritance_to_similarity"
    INTENSIONAL_INHERITANCE  = "intensional_inheritance"
    CONTEXT                  = "context"


# =========================================================================
# 2. Data Structures
# =========================================================================

@dataclass
class InferenceNode:
    """One node in a proof tree."""
    target_id:   str
    target_type: Optional[AtomType]  = None
    rule_used:   Optional[RuleType]  = None
    tv:          TruthValue          = field(default_factory=TruthValue.DEFAULT)
    children:    List["InferenceNode"] = field(default_factory=list)
    depth:       int                 = 0


@dataclass
class InferenceResult:
    """Result of a single inference query."""
    success:        bool                     = False
    target_id:      str                      = ""
    conclusion_tv:  Optional[TruthValue]     = None
    proof_tree:     Optional[InferenceNode]  = None
    steps_used:     int                      = 0
    rules_applied:  List[RuleType]           = field(default_factory=list)


# =========================================================================
# 3. Configuration
# =========================================================================

DEFAULT_CONFIDENCE_FACTORS: Dict[str, float] = {
    "deduction":                0.9,
    "induction":                0.6,
    "abduction":                0.5,
    "modus_ponens":             0.85,
    "revision":                 1.0,
    "and_rule":                 0.95,
    "or_rule":                  0.95,
    "not_rule":                 0.95,
    "similarity_to_inheritance": 0.7,
    "inheritance_to_similarity": 0.7,
    "intensional_inheritance":  0.5,
    "context":                  0.8,
}


@dataclass
class PLNConfig:
    max_depth:          int   = 5
    max_steps:          int   = 100
    min_confidence:     float = 0.1
    prior_strength:     float = 0.5
    prior_confidence:   float = 0.1

    confidence_factors: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CONFIDENCE_FACTORS),
    )

    mode_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "ANALYTICAL": {"min_confidence": 0.2, "max_depth": 4},
        "CREATIVE":   {"min_confidence": 0.05, "max_depth": 7},
        "REM_DREAM":  {"min_confidence": 0.01, "max_depth": 10},
        "DEFAULT":    {},
    })


# =========================================================================
# 4. Neurochemical Output
# =========================================================================

@dataclass
class PLNNeurochem:
    da_delta:     float = 0.0
    ach_delta:    float = 0.0
    ne_delta:     float = 0.0
    _5ht_delta:   float = 0.0
    gamma_boost:  float = 0.0
    theta_boost:  float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "da_delta":    self.da_delta,
            "ach_delta":   self.ach_delta,
            "ne_delta":    self.ne_delta,
            "_5ht_delta":  self._5ht_delta,
            "gamma_boost": self.gamma_boost,
            "theta_boost": self.theta_boost,
        }


# =========================================================================
# 5. NT Modulation Weights
# =========================================================================

_W_5HT  = 0.4   # Confidence conservatism
_W_DA   = 0.3   # Boldness
_W_NE   = 0.3   # Depth push
_W_ACH  = 0.3   # Rule precision
_W_GABA = 0.4   # Pruning
_W_COR  = 0.3   # Risk aversion

_EPS = 1e-10     # Small epsilon to prevent division by zero


# =========================================================================
# 6. Pure Inference Rule Functions
# =========================================================================

def rule_deduction(
    tv_ab: TruthValue,
    tv_bc: TruthValue,
    s_c: float = 0.5,
    cf: float = 0.9,
) -> TruthValue:
    """Deduction: (A→B, B→C) → A→C."""
    denom = max(1.0 - tv_ab.strength, _EPS)
    s_ac = tv_ab.strength * tv_bc.strength + (1.0 - tv_ab.strength) * (s_c - tv_ab.strength * tv_bc.strength) / denom
    s_ac = _clamp(s_ac)
    c_ac = min(tv_ab.confidence, tv_bc.confidence) * cf
    return TruthValue(strength=s_ac, confidence=_clamp(c_ac))


def rule_induction(
    tv_ab: TruthValue,
    tv_ac: TruthValue,
    s_c: float = 0.5,
    cf: float = 0.6,
) -> TruthValue:
    """Induction: (A→B, A→C) → B→C (analogical)."""
    s_bc = tv_ab.strength * tv_ac.strength + (1.0 - tv_ab.strength) * s_c
    s_bc = _clamp(s_bc)
    c_bc = min(tv_ab.confidence, tv_ac.confidence) * cf
    return TruthValue(strength=s_bc, confidence=_clamp(c_bc))


def rule_abduction(
    tv_ab: TruthValue,
    tv_cb: TruthValue,
    s_a: float = 0.5,
    s_b: float = 0.5,
    cf: float = 0.5,
) -> TruthValue:
    """Abduction: (A→B, C→B) → A→C (reverse)."""
    s_ac = tv_ab.strength * tv_cb.strength * s_a / max(s_b, _EPS)
    s_ac = _clamp(s_ac)
    c_ac = min(tv_ab.confidence, tv_cb.confidence) * cf
    return TruthValue(strength=s_ac, confidence=_clamp(c_ac))


def rule_modus_ponens(
    tv_a: TruthValue,
    tv_ab: TruthValue,
    s_b_prior: float = 0.5,
    cf: float = 0.85,
) -> TruthValue:
    """Modus ponens: (A, A→B) → B."""
    s_b = tv_a.strength * tv_ab.strength + (1.0 - tv_a.strength) * s_b_prior
    s_b = _clamp(s_b)
    c_b = min(tv_a.confidence, tv_ab.confidence) * cf
    return TruthValue(strength=s_b, confidence=_clamp(c_b))


def rule_revision(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """Revision: merge two truth values for the same atom."""
    w1, w2 = tv1.confidence, tv2.confidence
    w_total = w1 + w2
    if w_total < _EPS:
        return TruthValue(strength=0.5, confidence=0.0)
    s = (tv1.strength * w1 + tv2.strength * w2) / w_total
    c = min(w_total, 1.0)
    return TruthValue(strength=_clamp(s), confidence=_clamp(c))


def rule_and(
    tv_a: TruthValue,
    tv_b: TruthValue,
    cf: float = 0.95,
) -> TruthValue:
    """AND: (A, B) → A ∧ B."""
    s = tv_a.strength * tv_b.strength
    c = min(tv_a.confidence, tv_b.confidence) * cf
    return TruthValue(strength=_clamp(s), confidence=_clamp(c))


def rule_or(
    tv_a: TruthValue,
    tv_b: TruthValue,
    cf: float = 0.95,
) -> TruthValue:
    """OR: (A, B) → A ∨ B."""
    s = tv_a.strength + tv_b.strength - tv_a.strength * tv_b.strength
    c = min(tv_a.confidence, tv_b.confidence) * cf
    return TruthValue(strength=_clamp(s), confidence=_clamp(c))


def rule_not(tv_a: TruthValue, cf: float = 0.95) -> TruthValue:
    """NOT: (A) → ¬A."""
    s = 1.0 - tv_a.strength
    c = tv_a.confidence * cf
    return TruthValue(strength=_clamp(s), confidence=_clamp(c))


def rule_similarity_to_inheritance(
    tv_sim: TruthValue,
    cf: float = 0.7,
) -> TruthValue:
    """Convert Similarity → Inheritance."""
    return TruthValue(strength=tv_sim.strength, confidence=tv_sim.confidence * cf)


def rule_inheritance_to_similarity(
    tv_ab: TruthValue,
    tv_ba: TruthValue,
    cf: float = 0.7,
) -> TruthValue:
    """Convert Inheritance(A,B) + Inheritance(B,A) → Similarity(A,B)."""
    s = (tv_ab.strength + tv_ba.strength) / 2.0
    c = min(tv_ab.confidence, tv_ba.confidence) * cf
    return TruthValue(strength=_clamp(s), confidence=_clamp(c))


def rule_intensional_inheritance(
    shared_properties: int,
    total_properties: int,
    cf: float = 0.5,
) -> TruthValue:
    """Intensional inheritance based on shared properties."""
    if total_properties == 0:
        return TruthValue(strength=0.0, confidence=0.0)
    s = shared_properties / total_properties
    # Confidence from sample size (simple heuristic)
    c = min(total_properties / 20.0, 1.0) * cf
    return TruthValue(strength=_clamp(s), confidence=_clamp(c))


def rule_context(
    tv_ctx: TruthValue,
    tv_ab: TruthValue,
    cf: float = 0.8,
) -> TruthValue:
    """Context restriction: inference valid within a context."""
    return TruthValue(
        strength=tv_ab.strength,
        confidence=min(tv_ctx.confidence, tv_ab.confidence) * cf,
    )


# =========================================================================
# 7. Helper: Neurochem Computation
# =========================================================================

def compute_pln_neurochem(
    novel_inferences: int,
    successful_chains: int,
    failed_inferences: int,
    revisions: int,
    max_chain_depth: int,
) -> PLNNeurochem:
    """Compute NT output from inference cycle events."""
    nc = PLNNeurochem()
    nc.da_delta    = novel_inferences * 0.05
    nc.ach_delta   = successful_chains * 0.03
    nc.ne_delta    = failed_inferences * 0.04
    nc._5ht_delta  = revisions * 0.03
    nc.gamma_boost = successful_chains * 0.01
    if max_chain_depth >= 3:
        nc.theta_boost = 0.1
    return nc


# =========================================================================
# 8. PLN Engine
# =========================================================================

class PLNEngine:
    """
    Engine 10 — Probabilistic Logic Networks.

    Performs backward-chaining inference over an AtomSpace (E9),
    applying 12 inference rules with truth-value propagation.
    """

    engine_id: str = "pln_engine"
    cluster:   str = "knowledge_substrate"

    def __init__(
        self,
        atomspace: AtomSpaceEngine,
        config: Optional[PLNConfig] = None,
    ) -> None:
        self.atomspace = atomspace
        self.config = config or PLNConfig()

        # NT levels (Pattern A)
        self._5ht_level:  float = 0.5
        self.da_level:    float = 0.5
        self.ne_level:    float = 0.5
        self.ach_level:   float = 0.5
        self.gaba_level:  float = 0.5
        self.cor_level:   float = 0.5

        # Effective config (mode + NT modulated)
        self._eff_max_depth:      int   = self.config.max_depth
        self._eff_max_steps:      int   = self.config.max_steps
        self._eff_min_confidence: float = self.config.min_confidence

        # Mode
        self._mode: str = "DEFAULT"

        # Cycle counters
        self._tick_counter:       int = 0
        self._novel_inferences:   int = 0
        self._successful_chains:  int = 0
        self._failed_inferences:  int = 0
        self._revisions:          int = 0
        self._max_chain_depth:    int = 0

    # -----------------------------------------------------------------
    # Pattern A: Neurochemical State Update
    # -----------------------------------------------------------------

    def update_neurochem_state(self, nt_state: Dict[str, float]) -> None:
        self._5ht_level = _clamp(nt_state.get("5ht", self._5ht_level))
        self.da_level   = _clamp(nt_state.get("da",  self.da_level))
        self.ne_level   = _clamp(nt_state.get("ne",  self.ne_level))
        self.ach_level  = _clamp(nt_state.get("ach", self.ach_level))
        self.gaba_level = _clamp(nt_state.get("gaba", self.gaba_level))
        self.cor_level  = _clamp(nt_state.get("cor", self.cor_level))
        self._recompute_effective_params()

    def _recompute_effective_params(self) -> None:
        """Recompute NT-modulated effective parameters."""
        base_min_c = self.config.min_confidence
        mode_overrides = self.config.mode_configs.get(self._mode, {})
        base_min_c = mode_overrides.get("min_confidence", base_min_c)
        base_depth = mode_overrides.get("max_depth", self.config.max_depth)

        # 5-HT raises min confidence (conservative)
        # DA lowers min confidence (bold)
        self._eff_min_confidence = (
            base_min_c
            * (1.0 + _W_5HT * self._5ht_level)
            * max(1.0 - _W_DA * self.da_level, 0.1)
        )

        # NE increases depth (up to +5 at max NE)
        self._eff_max_depth = int(base_depth + _W_NE * self.ne_level * 5)

        # GABA reduces step budget
        self._eff_max_steps = int(
            self.config.max_steps
            * max(1.0 - _W_GABA * self.gaba_level * 0.5, 0.3)
        )

    # -----------------------------------------------------------------
    # Mode Configuration
    # -----------------------------------------------------------------

    def _apply_mode_config(self, mode: str) -> None:
        self._mode = mode
        self._recompute_effective_params()

    # -----------------------------------------------------------------
    # Core: Inference
    # -----------------------------------------------------------------

    def infer(
        self,
        target_id: str,
        max_depth: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> InferenceResult:
        """
        Backward-chain to find support for *target_id*.
        Returns InferenceResult with proof tree.
        """
        md = max_depth if max_depth is not None else self._eff_max_depth
        ms = max_steps if max_steps is not None else self._eff_max_steps
        mc = min_confidence if min_confidence is not None else self._eff_min_confidence

        step_counter = [0]  # Mutable counter shared across recursion
        rules_used: List[RuleType] = []

        tree = self._backward_chain(target_id, md, ms, mc, step_counter, rules_used, set())

        if tree is not None and tree.tv.confidence >= mc:
            self._successful_chains += 1
            self._novel_inferences += 1
            self._max_chain_depth = max(self._max_chain_depth, self._tree_depth(tree))
            return InferenceResult(
                success=True,
                target_id=target_id,
                conclusion_tv=tree.tv,
                proof_tree=tree,
                steps_used=step_counter[0],
                rules_applied=rules_used,
            )
        else:
            self._failed_inferences += 1
            return InferenceResult(
                success=False,
                target_id=target_id,
                conclusion_tv=None,
                proof_tree=tree,
                steps_used=step_counter[0],
                rules_applied=rules_used,
            )

    def infer_link(
        self,
        link_type: AtomType,
        source_id: str,
        target_id: str,
        max_depth: Optional[int] = None,
    ) -> InferenceResult:
        """Convenience: infer a specific link between two atoms."""
        # Check if such a link already exists — O(k) via directional index
        links_from_source = self.atomspace.get_links_from(source_id, link_type)
        for link in links_from_source:
            if link.outgoing == (source_id, target_id):
                return InferenceResult(
                    success=True,
                    target_id=link.atom_id,
                    conclusion_tv=link.truth_value,
                    proof_tree=InferenceNode(
                        target_id=link.atom_id,
                        target_type=link_type,
                        tv=link.truth_value,
                    ),
                    steps_used=0,
                    rules_applied=[],
                )

        # Try deduction: find B such that source→B and B→target
        md = max_depth if max_depth is not None else self._eff_max_depth
        mc = self._eff_min_confidence

        best_result: Optional[InferenceResult] = None

        if link_type == AtomType.INHERITANCE_LINK:
            best_result = self._try_deduction_for_link(source_id, target_id, md, mc)

        if best_result is not None and best_result.success:
            self._successful_chains += 1
            self._novel_inferences += 1
            return best_result

        self._failed_inferences += 1
        return InferenceResult(success=False, target_id=f"{source_id}->{target_id}")

    def check_consistency(
        self,
        atom_id_a: str,
        atom_id_b: str,
    ) -> Dict[str, Any]:
        """
        Check if two atoms are consistent (no contradiction).
        Returns dict with 'consistent', 'confidence', 'detail'.
        """
        atom_a = self.atomspace.get_atom(atom_id_a)
        atom_b = self.atomspace.get_atom(atom_id_b)

        if atom_a is None or atom_b is None:
            return {"consistent": True, "confidence": 0.0, "detail": "atom_not_found"}

        # Check if there's a NOT link negating atom A — O(k) via directional index
        not_links_on_a = self.atomspace.get_links_from(atom_id_a, AtomType.NOT_LINK)
        for nl in not_links_on_a:
            if len(nl.outgoing) == 1:
                negated_id = nl.outgoing[0]
                if negated_id == atom_id_a:
                    # ¬A exists, check if B implies A
                    result = self.infer_link(AtomType.IMPLICATION_LINK, atom_id_b, atom_id_a, max_depth=3)
                    if result.success and result.conclusion_tv and result.conclusion_tv.strength > 0.5:
                        return {
                            "consistent": False,
                            "confidence": result.conclusion_tv.confidence,
                            "detail": "b_implies_negated_a",
                        }

        return {"consistent": True, "confidence": 0.5, "detail": "no_contradiction_found"}

    # -----------------------------------------------------------------
    # Internal: Backward Chaining
    # -----------------------------------------------------------------

    def _backward_chain(
        self,
        target_id: str,
        max_depth: int,
        max_steps: int,
        min_confidence: float,
        step_counter: List[int],
        rules_used: List[RuleType],
        visited: set,
    ) -> Optional[InferenceNode]:
        """Recursive backward chaining."""
        # Budget check
        if max_depth <= 0 or step_counter[0] >= max_steps:
            return None

        # Cycle detection
        if target_id in visited:
            return None
        visited = visited | {target_id}

        # Step 1: Check if target is already in AtomSpace with sufficient TV
        existing = self.atomspace.get_atom(target_id)
        if existing is not None and existing.truth_value.confidence >= min_confidence:
            return InferenceNode(
                target_id=target_id,
                target_type=existing.atom_type,
                tv=existing.truth_value,
                depth=0,
            )

        if existing is None:
            return None  # Can't reason about unknown atoms

        # Step 2: Try applicable rules
        step_counter[0] += 1
        best_node: Optional[InferenceNode] = None

        # Try revision first (if multiple TVs exist for same concept)
        revision_node = self._try_revision(existing, min_confidence, rules_used)
        if revision_node is not None:
            best_node = revision_node

        # Try modus ponens: find A and A→target
        mp_node = self._try_modus_ponens(
            target_id, existing, max_depth - 1, max_steps,
            min_confidence, step_counter, rules_used, visited,
        )
        if mp_node is not None:
            if best_node is None or mp_node.tv.confidence > best_node.tv.confidence:
                best_node = mp_node

        # Try deduction for links: if target is a link, find intermediate
        if existing.atom_type in {AtomType.INHERITANCE_LINK, AtomType.IMPLICATION_LINK}:
            deduction_node = self._try_deduction(
                existing, max_depth - 1, max_steps,
                min_confidence, step_counter, rules_used, visited,
            )
            if deduction_node is not None:
                if best_node is None or deduction_node.tv.confidence > best_node.tv.confidence:
                    best_node = deduction_node

        # Try NOT rule for NotLinks
        if existing.atom_type == AtomType.NOT_LINK and len(existing.outgoing) == 1:
            not_node = self._try_not(
                existing, max_depth - 1, max_steps,
                min_confidence, step_counter, rules_used, visited,
            )
            if not_node is not None:
                if best_node is None or not_node.tv.confidence > best_node.tv.confidence:
                    best_node = not_node

        # Try AND for AndLinks
        if existing.atom_type == AtomType.AND_LINK:
            and_node = self._try_and(
                existing, max_depth - 1, max_steps,
                min_confidence, step_counter, rules_used, visited,
            )
            if and_node is not None:
                if best_node is None or and_node.tv.confidence > best_node.tv.confidence:
                    best_node = and_node

        # Try OR for OrLinks
        if existing.atom_type == AtomType.OR_LINK:
            or_node = self._try_or(
                existing, max_depth - 1, max_steps,
                min_confidence, step_counter, rules_used, visited,
            )
            if or_node is not None:
                if best_node is None or or_node.tv.confidence > best_node.tv.confidence:
                    best_node = or_node

        return best_node

    def _try_revision(
        self,
        atom: Atom,
        min_confidence: float,
        rules_used: List[RuleType],
    ) -> Optional[InferenceNode]:
        """Try combining multiple truth values for the same atom via revision."""
        # Look for atoms with same name + type (different sources)
        if atom.name is None:
            return None
        candidates = self.atomspace.get_by_name(atom.name)
        if len(candidates) < 2:
            return None

        combined_tv = atom.truth_value
        children: List[InferenceNode] = []
        for candidate in candidates:
            if candidate.atom_id != atom.atom_id and candidate.atom_type == atom.atom_type:
                combined_tv = rule_revision(combined_tv, candidate.truth_value)
                children.append(InferenceNode(
                    target_id=candidate.atom_id,
                    target_type=candidate.atom_type,
                    tv=candidate.truth_value,
                ))
                self._revisions += 1

        if children and combined_tv.confidence >= min_confidence:
            rules_used.append(RuleType.REVISION)
            return InferenceNode(
                target_id=atom.atom_id,
                target_type=atom.atom_type,
                rule_used=RuleType.REVISION,
                tv=combined_tv,
                children=children,
                depth=1,
            )
        return None

    def _try_modus_ponens(
        self,
        target_id: str,
        target_atom: Atom,
        max_depth: int,
        max_steps: int,
        min_confidence: float,
        step_counter: List[int],
        rules_used: List[RuleType],
        visited: set,
    ) -> Optional[InferenceNode]:
        """Modus ponens: find A and A→target."""
        # O(k) lookup via directional index: all impl links pointing TO target
        impl_links = self.atomspace.get_links_to(target_id, AtomType.IMPLICATION_LINK)
        best: Optional[InferenceNode] = None

        for link in impl_links:
            if step_counter[0] >= max_steps:
                break
            if len(link.outgoing) != 2:
                continue

            antecedent_id = link.outgoing[0]
            antecedent = self.atomspace.get_atom(antecedent_id)
            if antecedent is None:
                continue

            # Recursively prove the antecedent
            ante_node = self._backward_chain(
                antecedent_id, max_depth, max_steps,
                min_confidence, step_counter, rules_used, visited,
            )
            if ante_node is None:
                continue

            cf = self.config.confidence_factors.get("modus_ponens", 0.85)
            result_tv = rule_modus_ponens(
                ante_node.tv, link.truth_value,
                s_b_prior=self.config.prior_strength,
                cf=cf,
            )

            if result_tv.confidence >= min_confidence:
                node = InferenceNode(
                    target_id=target_id,
                    target_type=target_atom.atom_type,
                    rule_used=RuleType.MODUS_PONENS,
                    tv=result_tv,
                    children=[
                        ante_node,
                        InferenceNode(
                            target_id=link.atom_id,
                            target_type=AtomType.IMPLICATION_LINK,
                            tv=link.truth_value,
                        ),
                    ],
                    depth=ante_node.depth + 1,
                )
                rules_used.append(RuleType.MODUS_PONENS)
                if best is None or result_tv.confidence > best.tv.confidence:
                    best = node

        return best

    def _try_deduction(
        self,
        link_atom: Atom,
        max_depth: int,
        max_steps: int,
        min_confidence: float,
        step_counter: List[int],
        rules_used: List[RuleType],
        visited: set,
    ) -> Optional[InferenceNode]:
        """Deduction: for A→C link, find B s.t. A→B and B→C."""
        if len(link_atom.outgoing) != 2:
            return None

        source_id, target_id = link_atom.outgoing
        link_type = link_atom.atom_type

        return self._try_deduction_for_link_internal(
            source_id, target_id, link_type,
            max_depth, max_steps, min_confidence,
            step_counter, rules_used, visited,
        )

    def _try_deduction_for_link(
        self,
        source_id: str,
        target_id: str,
        max_depth: int,
        min_confidence: float,
    ) -> Optional[InferenceResult]:
        """Try deduction for an InheritanceLink between source→target."""
        step_counter = [0]
        rules_used: List[RuleType] = []
        node = self._try_deduction_for_link_internal(
            source_id, target_id, AtomType.INHERITANCE_LINK,
            max_depth, self._eff_max_steps, min_confidence,
            step_counter, rules_used, set(),
        )
        if node is not None and node.tv.confidence >= min_confidence:
            return InferenceResult(
                success=True,
                target_id=f"{source_id}->{target_id}",
                conclusion_tv=node.tv,
                proof_tree=node,
                steps_used=step_counter[0],
                rules_applied=rules_used,
            )
        return None

    def _try_deduction_for_link_internal(
        self,
        source_id: str,
        target_id: str,
        link_type: AtomType,
        max_depth: int,
        max_steps: int,
        min_confidence: float,
        step_counter: List[int],
        rules_used: List[RuleType],
        visited: set,
    ) -> Optional[InferenceNode]:
        """Find B such that source→B and B→target.

        Uses directional indices for O(k₁ + k₂) instead of O(n²).
        """
        if step_counter[0] >= max_steps or max_depth <= 0:
            return None

        step_counter[0] += 1

        # O(k₁): links FROM source of this type
        ab_links = self.atomspace.get_links_from(source_id, link_type)
        # O(k₂): links TO target of this type
        bc_links = self.atomspace.get_links_to(target_id, link_type)

        # Build a fast set of B candidates that connect TO target
        # b_id → bc_link for O(1) lookup
        b_to_target: Dict[str, Atom] = {}
        for bc_link in bc_links:
            if len(bc_link.outgoing) == 2:
                b_to_target[bc_link.outgoing[0]] = bc_link

        best: Optional[InferenceNode] = None

        for ab_link in ab_links:
            if len(ab_link.outgoing) != 2:
                continue

            b_id = ab_link.outgoing[1]
            if b_id == target_id or b_id == source_id:
                continue  # Not an intermediate step / self-loop

            # O(1): check if B connects to target
            bc_link = b_to_target.get(b_id)
            if bc_link is None:
                continue

            cf = self.config.confidence_factors.get("deduction", 0.9)
            result_tv = rule_deduction(
                ab_link.truth_value,
                bc_link.truth_value,
                s_c=self.config.prior_strength,
                cf=cf,
            )

            if result_tv.confidence >= min_confidence:
                node = InferenceNode(
                    target_id=f"{source_id}->{target_id}",
                    target_type=link_type,
                    rule_used=RuleType.DEDUCTION,
                    tv=result_tv,
                    children=[
                        InferenceNode(
                            target_id=ab_link.atom_id,
                            target_type=link_type,
                            tv=ab_link.truth_value,
                        ),
                        InferenceNode(
                            target_id=bc_link.atom_id,
                            target_type=link_type,
                            tv=bc_link.truth_value,
                        ),
                    ],
                    depth=1,
                )
                rules_used.append(RuleType.DEDUCTION)
                if best is None or result_tv.confidence > best.tv.confidence:
                    best = node

        return best

    def _try_not(
        self, atom: Atom, max_depth: int, max_steps: int,
        min_confidence: float, step_counter: List[int],
        rules_used: List[RuleType], visited: set,
    ) -> Optional[InferenceNode]:
        """NOT: infer ¬A from A."""
        inner_id = atom.outgoing[0]
        inner_node = self._backward_chain(
            inner_id, max_depth, max_steps,
            min_confidence, step_counter, rules_used, visited,
        )
        if inner_node is None:
            return None
        cf = self.config.confidence_factors.get("not_rule", 0.95)
        tv = rule_not(inner_node.tv, cf=cf)
        if tv.confidence >= min_confidence:
            rules_used.append(RuleType.NOT)
            return InferenceNode(
                target_id=atom.atom_id,
                target_type=AtomType.NOT_LINK,
                rule_used=RuleType.NOT,
                tv=tv,
                children=[inner_node],
                depth=inner_node.depth + 1,
            )
        return None

    def _try_and(
        self, atom: Atom, max_depth: int, max_steps: int,
        min_confidence: float, step_counter: List[int],
        rules_used: List[RuleType], visited: set,
    ) -> Optional[InferenceNode]:
        """AND: infer A∧B from A and B."""
        if len(atom.outgoing) < 2:
            return None
        children: List[InferenceNode] = []
        for out_id in atom.outgoing:
            child = self._backward_chain(
                out_id, max_depth, max_steps,
                min_confidence, step_counter, rules_used, visited,
            )
            if child is None:
                return None
            children.append(child)

        # Chain AND across all children
        combined_tv = children[0].tv
        cf = self.config.confidence_factors.get("and_rule", 0.95)
        for i in range(1, len(children)):
            combined_tv = rule_and(combined_tv, children[i].tv, cf=cf)

        if combined_tv.confidence >= min_confidence:
            rules_used.append(RuleType.AND)
            max_d = max(c.depth for c in children)
            return InferenceNode(
                target_id=atom.atom_id,
                target_type=AtomType.AND_LINK,
                rule_used=RuleType.AND,
                tv=combined_tv,
                children=children,
                depth=max_d + 1,
            )
        return None

    def _try_or(
        self, atom: Atom, max_depth: int, max_steps: int,
        min_confidence: float, step_counter: List[int],
        rules_used: List[RuleType], visited: set,
    ) -> Optional[InferenceNode]:
        """OR: infer A∨B — succeed if at least one child is provable."""
        if len(atom.outgoing) < 2:
            return None
        children: List[InferenceNode] = []
        for out_id in atom.outgoing:
            child = self._backward_chain(
                out_id, max_depth, max_steps,
                min_confidence, step_counter, rules_used, visited,
            )
            if child is not None:
                children.append(child)
            else:
                # Unknown child → treat as prior
                children.append(InferenceNode(
                    target_id=out_id,
                    tv=TruthValue(self.config.prior_strength, self.config.prior_confidence),
                ))

        combined_tv = children[0].tv
        cf = self.config.confidence_factors.get("or_rule", 0.95)
        for i in range(1, len(children)):
            combined_tv = rule_or(combined_tv, children[i].tv, cf=cf)

        if combined_tv.confidence >= min_confidence:
            rules_used.append(RuleType.OR)
            max_d = max(c.depth for c in children)
            return InferenceNode(
                target_id=atom.atom_id,
                target_type=AtomType.OR_LINK,
                rule_used=RuleType.OR,
                tv=combined_tv,
                children=children,
                depth=max_d + 1,
            )
        return None

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _tree_depth(node: InferenceNode) -> int:
        if not node.children:
            return 0
        return 1 + max(PLNEngine._tree_depth(c) for c in node.children)

    # -----------------------------------------------------------------
    # process() — Main Entry Point
    # -----------------------------------------------------------------

    def process(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        input_data = input_data or {}

        # Reset cycle counters
        self._novel_inferences = 0
        self._successful_chains = 0
        self._failed_inferences = 0
        self._revisions = 0
        self._max_chain_depth = 0

        # 1. NT state
        if "nt_state" in input_data:
            self.update_neurochem_state(input_data["nt_state"])

        # 2. Mode
        if "mode" in input_data:
            self._apply_mode_config(input_data["mode"])

        # 3. Execute queries
        results: List[Dict[str, Any]] = []
        for query in input_data.get("queries", []):
            target_id = query.get("target_id", "")
            result = self.infer(
                target_id,
                max_depth=query.get("max_depth"),
                max_steps=query.get("max_steps"),
                min_confidence=query.get("min_confidence"),
            )
            results.append({
                "success":      result.success,
                "target_id":    result.target_id,
                "strength":     result.conclusion_tv.strength if result.conclusion_tv else None,
                "confidence":   result.conclusion_tv.confidence if result.conclusion_tv else None,
                "steps_used":   result.steps_used,
                "rules_applied": [r.value for r in result.rules_applied],
            })

        self._tick_counter += 1

        # 4. Neurochem
        signals = compute_pln_neurochem(
            self._novel_inferences,
            self._successful_chains,
            self._failed_inferences,
            self._revisions,
            self._max_chain_depth,
        )

        return {
            "results":           results,
            "neurochem_signals": signals.as_dict(),
            "stats": {
                "novel_inferences":  self._novel_inferences,
                "successful_chains": self._successful_chains,
                "failed_inferences": self._failed_inferences,
                "revisions":         self._revisions,
                "max_chain_depth":   self._max_chain_depth,
            },
        }

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id":        self.engine_id,
            "cluster":          self.cluster,
            "mode":             self._mode,
            "tick_counter":     self._tick_counter,
            "eff_max_depth":    self._eff_max_depth,
            "eff_max_steps":    self._eff_max_steps,
            "eff_min_confidence": self._eff_min_confidence,
            "nt_levels": {
                "5ht": self._5ht_level,
                "da":  self.da_level,
                "ne":  self.ne_level,
                "ach": self.ach_level,
                "gaba": self.gaba_level,
                "cor": self.cor_level,
            },
        }

    def __repr__(self) -> str:
        return f"PLNEngine(mode={self._mode}, depth={self._eff_max_depth})"
