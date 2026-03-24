"""
Engine 9 — AtomSpace-Lite  (Typed Hypergraph Knowledge Store)
=============================================================

Pure-Python reimplementation of the **algorithmic core** of OpenCog's
AtomSpace.  Every piece of knowledge is an *Atom* — either a *Node*
(concept, predicate, number …) or a *Link* (typed relationship between
atoms).  Each atom carries a TruthValue (strength, confidence) enabling
uncertain knowledge, and an AttentionValue (STI, LTI) managed by the
ECAN engine (E16).

Key differences from full OpenCog AtomSpace:
  • ~15 atom types (not hundreds)
  • Simple Truth Values only (no Indefinite/Count/Fuzzy)
  • Neurochemical modulation of write-gate, decay, pattern matching
  • Integrated with ZADOS pipeline via Pattern A ``update_neurochem_state``
  • No MeTTa interpreter — Python is the meta-language

Spec: ``docs/specs/engine_9_atomspace_spec.md``
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from zados.cognitive_engines.constants import _clamp

# =========================================================================
# 1.  Atom Type Hierarchy
# =========================================================================

class AtomType(str, Enum):
    # --- Node types ---
    CONCEPT_NODE   = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    NUMBER_NODE    = "NumberNode"
    VARIABLE_NODE  = "VariableNode"
    SCHEMA_NODE    = "SchemaNode"
    GROUNDED_NODE  = "GroundedNode"

    # --- Link types ---
    INHERITANCE_LINK = "InheritanceLink"
    SIMILARITY_LINK  = "SimilarityLink"
    EVALUATION_LINK  = "EvaluationLink"
    LIST_LINK        = "ListLink"
    AND_LINK         = "AndLink"
    OR_LINK          = "OrLink"
    NOT_LINK         = "NotLink"
    IMPLICATION_LINK = "ImplicationLink"
    HEBBIAN_LINK     = "HebbianLink"


NODE_TYPES: FrozenSet[AtomType] = frozenset({
    AtomType.CONCEPT_NODE,
    AtomType.PREDICATE_NODE,
    AtomType.NUMBER_NODE,
    AtomType.VARIABLE_NODE,
    AtomType.SCHEMA_NODE,
    AtomType.GROUNDED_NODE,
})

LINK_TYPES: FrozenSet[AtomType] = frozenset({
    AtomType.INHERITANCE_LINK,
    AtomType.SIMILARITY_LINK,
    AtomType.EVALUATION_LINK,
    AtomType.LIST_LINK,
    AtomType.AND_LINK,
    AtomType.OR_LINK,
    AtomType.NOT_LINK,
    AtomType.IMPLICATION_LINK,
    AtomType.HEBBIAN_LINK,
})

SYMMETRIC_LINKS: FrozenSet[AtomType] = frozenset({
    AtomType.SIMILARITY_LINK,
    AtomType.HEBBIAN_LINK,
})


def is_node_type(t: AtomType) -> bool:
    return t in NODE_TYPES


def is_link_type(t: AtomType) -> bool:
    return t in LINK_TYPES


# =========================================================================
# 2.  TruthValue
# =========================================================================

@dataclass(frozen=True)
class TruthValue:
    """Simple Truth Value (STV): (strength, confidence)."""
    strength:   float = 1.0   # s ∈ [0, 1]
    confidence: float = 0.0   # c ∈ [0, 1]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strength",   _clamp(self.strength))
        object.__setattr__(self, "confidence", _clamp(self.confidence))

    # -- Factory helpers --------------------------------------------------
    @staticmethod
    def DEFAULT() -> TruthValue:
        return TruthValue(strength=1.0, confidence=0.0)

    @staticmethod
    def TRUE() -> TruthValue:
        return TruthValue(strength=1.0, confidence=0.9)

    @staticmethod
    def FALSE() -> TruthValue:
        return TruthValue(strength=0.0, confidence=0.9)


# =========================================================================
# 3.  AttentionValue  (STI / LTI — owned by ECAN, stored here)
# =========================================================================

@dataclass
class AttentionValue:
    sti: float = 0.0   # Short-Term Importance
    lti: float = 0.0   # Long-Term Importance


# =========================================================================
# 4.  Atom
# =========================================================================

@dataclass
class Atom:
    atom_id:          str
    atom_type:        AtomType
    name:             Optional[str]           = None
    outgoing:         Tuple[str, ...]         = ()
    truth_value:      TruthValue              = field(default_factory=TruthValue.DEFAULT)
    attention_value:  AttentionValue           = field(default_factory=AttentionValue)
    incoming:         Set[str]                = field(default_factory=set)
    timetag:          int                     = 0
    source_engine:    Optional[str]           = None
    metadata:         Dict[str, Any]          = field(default_factory=dict)


# =========================================================================
# 5.  Pattern Matching Structures
# =========================================================================

@dataclass(frozen=True)
class PatternAtom:
    """Element of a match pattern — concrete reference *or* variable."""
    variable:   Optional[str]      = None    # "$X", "$Y"
    atom_type:  Optional[AtomType] = None    # type constraint
    name:       Optional[str]      = None    # name constraint (nodes)
    atom_id:    Optional[str]      = None    # exact atom id


@dataclass(frozen=True)
class Pattern:
    """Match template applied against links of *root_type*."""
    root_type:          AtomType
    elements:           Tuple[PatternAtom, ...]
    tv_min_strength:    float = 0.0
    tv_min_confidence:  float = 0.0


# =========================================================================
# 6.  Configuration
# =========================================================================

@dataclass
class AtomSpaceConfig:
    # Capacity
    max_atoms:              int   = 50_000
    max_incoming_per_atom:  int   = 1_000

    # Truth-value decay
    tv_decay_rate:          float = 0.001
    tv_decay_floor:         float = 0.05
    tv_identity_immune:     bool  = True

    # Pruning
    prune_threshold_tv:     float = 0.05
    prune_threshold_sti:    float = -100.0
    prune_protect_lti:      float = 0.5

    # Write gating
    min_confidence_to_add:  float = 0.1
    novelty_threshold:      float = 0.8

    # Pattern matching
    max_match_results:      int   = 100
    max_match_depth:        int   = 10

    # Maintenance interval (cycles between decay/prune)
    maintenance_interval:   int   = 10

    # Mode overrides
    mode_configs: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "ANALYTICAL": {
            "min_confidence_to_add": 0.2,
            "prune_threshold_tv":   0.1,
        },
        "CREATIVE": {
            "min_confidence_to_add": 0.05,
            "novelty_threshold":    0.9,
        },
        "REM_DREAM": {
            "min_confidence_to_add": 0.01,
            "novelty_threshold":    0.95,
            "prune_threshold_sti":  -200.0,
        },
        "DEFAULT": {},
    })


# =========================================================================
# 7.  Neurochemical Output
# =========================================================================

@dataclass
class AtomSpaceNeurochem:
    da_delta:        float = 0.0
    ne_delta:        float = 0.0
    ach_delta:       float = 0.0
    _5ht_delta:      float = 0.0
    # Oscillatory
    gamma_boost:     float = 0.0
    theta_boost:     float = 0.0
    alpha_suppress:  float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "da_delta":        self.da_delta,
            "ne_delta":        self.ne_delta,
            "ach_delta":       self.ach_delta,
            "_5ht_delta":      self._5ht_delta,
            "gamma_boost":     self.gamma_boost,
            "theta_boost":     self.theta_boost,
            "alpha_suppress":  self.alpha_suppress,
        }


# =========================================================================
# 8.  NT Modulation Weights
# =========================================================================

_W_5HT  = 0.4    # Decay stabilization
_W_DA   = 0.3    # Write gate relaxation
_W_NE   = 0.3    # Match scope broadening
_W_ACH  = 0.3    # Query precision tightening
_W_GABA = 0.4    # Prune acceleration
_W_COR  = 0.3    # Conservative gate tightening
_W_OXT  = 0.2    # Social atom protection
_W_CB1  = 0.3    # Creative relaxation


# =========================================================================
# 9.  Pure Helper Functions
# =========================================================================

def make_atom(
    atom_type:    AtomType,
    name:         Optional[str],
    outgoing:     Tuple[str, ...],
    tv:           TruthValue,
    av:           AttentionValue,
    timetag:      int,
    source_engine: Optional[str] = None,
    metadata:     Optional[Dict[str, Any]] = None,
) -> Atom:
    """Create a fresh Atom with a new UUID."""
    return Atom(
        atom_id=str(uuid.uuid4()),
        atom_type=atom_type,
        name=name,
        outgoing=outgoing,
        truth_value=tv,
        attention_value=av,
        incoming=set(),
        timetag=timetag,
        source_engine=source_engine,
        metadata=metadata if metadata is not None else {},
    )


def merge_truth_values(old: TruthValue, new: TruthValue) -> TruthValue:
    """PLN revision rule — combine two truth values."""
    w_old = old.confidence
    w_new = new.confidence
    w_total = w_old + w_new
    if w_total == 0.0:
        return TruthValue(strength=0.5, confidence=0.0)
    s_merged = (old.strength * w_old + new.strength * w_new) / w_total
    c_merged = min(w_total, 1.0)
    return TruthValue(strength=_clamp(s_merged), confidence=_clamp(c_merged))


def compute_tv_decay(
    tv: TruthValue,
    rate: float,
    floor: float,
) -> TruthValue:
    """Decay confidence by *rate*, floored at *floor*."""
    new_c = max(tv.confidence - rate, floor)
    return TruthValue(strength=tv.strength, confidence=new_c)


def compute_write_gate_threshold(
    base: float,
    cor: float,
    da: float,
    cb1: float,
) -> float:
    """NT-modulated minimum confidence for write admission."""
    t = base * (1.0 + _W_COR * cor) * max(1.0 - _W_DA * da, 0.1) * max(1.0 - _W_CB1 * cb1, 0.1)
    return max(t, 0.0)


def compute_similarity_score(name_a: Optional[str], name_b: Optional[str]) -> float:
    """Quick name-based similarity for duplicate detection."""
    if name_a is None or name_b is None:
        return 0.0
    if name_a == name_b:
        return 1.0
    # Jaccard on character trigrams
    def _trigrams(s: str) -> Set[str]:
        s = s.lower()
        if len(s) < 3:
            return {s}
        return {s[i:i + 3] for i in range(len(s) - 2)}
    a_set = _trigrams(name_a)
    b_set = _trigrams(name_b)
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def match_pattern_atom(
    pa: PatternAtom,
    candidate: Atom,
    bindings: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """
    Check if *candidate* satisfies *pa*.  Returns updated bindings or None.
    """
    # Exact atom_id constraint
    if pa.atom_id is not None:
        if candidate.atom_id != pa.atom_id:
            return None

    # Type constraint
    if pa.atom_type is not None:
        if candidate.atom_type != pa.atom_type:
            return None

    # Name constraint
    if pa.name is not None:
        if candidate.name != pa.name:
            return None

    # Variable binding
    if pa.variable is not None:
        existing = bindings.get(pa.variable)
        if existing is not None:
            if existing != candidate.atom_id:
                return None  # Binding inconsistency
        else:
            bindings = dict(bindings)  # Copy to avoid mutation
            bindings[pa.variable] = candidate.atom_id

    return bindings


def match_pattern(
    pattern: Pattern,
    link: Atom,
    atoms_by_id: Dict[str, Atom],
) -> Optional[Dict[str, str]]:
    """
    Match *pattern* against *link*.  Returns variable bindings or None.
    """
    # TV filter
    if link.truth_value.strength < pattern.tv_min_strength:
        return None
    if link.truth_value.confidence < pattern.tv_min_confidence:
        return None

    # Arity check
    if len(link.outgoing) != len(pattern.elements):
        return None

    bindings: Dict[str, str] = {}
    for pa, target_id in zip(pattern.elements, link.outgoing):
        target = atoms_by_id.get(target_id)
        if target is None:
            return None
        result = match_pattern_atom(pa, target, bindings)
        if result is None:
            return None
        bindings = result

    return bindings


def score_atom_for_pruning(atom: Atom) -> float:
    """Combined score: lower → prune first."""
    tv_score = atom.truth_value.confidence
    sti_score = max(atom.attention_value.sti, 0.0) * 0.01
    lti_score = max(atom.attention_value.lti, 0.0) * 0.1
    return tv_score + sti_score + lti_score


def compute_atomspace_neurochem(
    novel_atoms_added: int,
    atoms_merged: int,
    match_results: int,
    match_failures: int,
    atoms_pruned: int,
) -> AtomSpaceNeurochem:
    """Compute NT output signals from cycle events."""
    nc = AtomSpaceNeurochem()
    nc.da_delta       = novel_atoms_added * 0.05
    nc._5ht_delta     = atoms_merged * 0.03
    nc.ach_delta      = match_results * 0.02
    nc.ne_delta       = match_failures * 0.04
    nc.gamma_boost    = atoms_pruned * 0.01
    if atoms_merged > 10:
        nc.theta_boost = 0.1
    if novel_atoms_added > 5:
        nc.alpha_suppress = 0.05
    return nc


# =========================================================================
# 10. AtomSpace Engine
# =========================================================================

class AtomSpaceEngine:
    """
    Engine 9 — Typed Hypergraph Knowledge Store.

    Provides structured knowledge storage with typed nodes and links,
    truth values, pattern matching, and neurochemical modulation.
    Serves as the knowledge substrate for PLN (E10) and ECAN (E16).
    """

    engine_id: str = "atomspace_engine"
    cluster:   str = "knowledge_substrate"

    def __init__(self, config: Optional[AtomSpaceConfig] = None) -> None:
        self.config = config or AtomSpaceConfig()

        # Primary store
        self._atoms: Dict[str, Atom] = {}

        # Indices
        self._by_type:     Dict[AtomType, Set[str]]       = defaultdict(set)
        self._by_name:     Dict[str, Set[str]]             = defaultdict(set)
        self._by_outgoing: Dict[FrozenSet[str], Set[str]]  = defaultdict(set)
        # Directional link indices: (source_atom_id, link_type) → {link_ids}
        self._links_from:  Dict[Tuple[str, AtomType], Set[str]] = defaultdict(set)
        # (target_atom_id, link_type) → {link_ids}
        self._links_to:    Dict[Tuple[str, AtomType], Set[str]] = defaultdict(set)

        # Counters
        self._next_timetag: int = 0
        self._tick_counter: int = 0

        # NT levels (Pattern A)
        self._5ht_level:     float = 0.5
        self.da_level:       float = 0.5
        self.ne_level:       float = 0.5
        self.ach_level:      float = 0.5
        self.gaba_level:     float = 0.5
        self.cor_level:      float = 0.5
        self.oxt_level:      float = 0.5
        self.cb1_level:      float = 0.5

        # Current mode
        self._mode: str = "DEFAULT"

        # Effective config values (may be overridden by mode)
        self._eff_min_confidence_to_add: float = self.config.min_confidence_to_add
        self._eff_novelty_threshold:     float = self.config.novelty_threshold
        self._eff_prune_threshold_tv:    float = self.config.prune_threshold_tv
        self._eff_prune_threshold_sti:   float = self.config.prune_threshold_sti

        # Cycle event counters (reset each process() call)
        self._novel_atoms_added: int = 0
        self._atoms_merged:      int = 0
        self._match_results:     int = 0
        self._match_failures:    int = 0
        self._atoms_pruned:      int = 0

    # -----------------------------------------------------------------
    # Pattern A: Neurochemical State Update
    # -----------------------------------------------------------------

    def update_neurochem_state(self, nt_state: Dict[str, float]) -> None:
        """Accept NT concentrations from the pipeline orchestrator."""
        self._5ht_level = _clamp(nt_state.get("5ht", self._5ht_level))
        self.da_level   = _clamp(nt_state.get("da",  self.da_level))
        self.ne_level   = _clamp(nt_state.get("ne",  self.ne_level))
        self.ach_level  = _clamp(nt_state.get("ach", self.ach_level))
        self.gaba_level = _clamp(nt_state.get("gaba", self.gaba_level))
        self.cor_level  = _clamp(nt_state.get("cor", self.cor_level))
        self.oxt_level  = _clamp(nt_state.get("oxt", self.oxt_level))
        self.cb1_level  = _clamp(nt_state.get("cb1", self.cb1_level))

    # -----------------------------------------------------------------
    # Mode Configuration
    # -----------------------------------------------------------------

    def _apply_mode_config(self, mode: str) -> None:
        """Apply mode-dependent overrides to effective config."""
        self._mode = mode
        overrides = self.config.mode_configs.get(mode, {})
        self._eff_min_confidence_to_add = overrides.get(
            "min_confidence_to_add", self.config.min_confidence_to_add,
        )
        self._eff_novelty_threshold = overrides.get(
            "novelty_threshold", self.config.novelty_threshold,
        )
        self._eff_prune_threshold_tv = overrides.get(
            "prune_threshold_tv", self.config.prune_threshold_tv,
        )
        self._eff_prune_threshold_sti = overrides.get(
            "prune_threshold_sti", self.config.prune_threshold_sti,
        )

    # -----------------------------------------------------------------
    # Write Operations
    # -----------------------------------------------------------------

    def add_node(
        self,
        atom_type:     AtomType,
        name:          str,
        truth_value:   Optional[TruthValue] = None,
        source_engine: Optional[str]        = None,
        metadata:      Optional[Dict[str, Any]] = None,
    ) -> Atom:
        """Add a node.  Merges if a matching node already exists."""
        if atom_type not in NODE_TYPES:
            raise ValueError(f"{atom_type} is not a node type")

        tv = truth_value or TruthValue.DEFAULT()

        # Write gate
        threshold = compute_write_gate_threshold(
            self._eff_min_confidence_to_add,
            self.cor_level, self.da_level, self.cb1_level,
        )
        if tv.confidence < threshold:
            tv = TruthValue(strength=tv.strength, confidence=threshold)

        # Check for existing node with same type + name
        existing = self._find_existing_node(atom_type, name)
        if existing is not None:
            sim = compute_similarity_score(name, existing.name)
            if sim >= self._eff_novelty_threshold:
                # Merge
                merged_tv = merge_truth_values(existing.truth_value, tv)
                existing.truth_value = merged_tv
                if metadata:
                    existing.metadata.update(metadata)
                self._atoms_merged += 1
                return existing

        # Capacity check
        if len(self._atoms) >= self.config.max_atoms:
            removed = self.enforce_capacity()
            if len(self._atoms) >= self.config.max_atoms and removed == 0:
                raise RuntimeError("AtomSpace at capacity; cannot add atom")

        # Create new atom
        atom = make_atom(
            atom_type=atom_type,
            name=name,
            outgoing=(),
            tv=tv,
            av=AttentionValue(),
            timetag=self._next_timetag,
            source_engine=source_engine,
            metadata=metadata,
        )
        self._next_timetag += 1

        # Store + index
        self._atoms[atom.atom_id] = atom
        self._by_type[atom_type].add(atom.atom_id)
        self._by_name[name].add(atom.atom_id)

        self._novel_atoms_added += 1
        return atom

    def add_link(
        self,
        atom_type:     AtomType,
        outgoing_ids:  Tuple[str, ...],
        truth_value:   Optional[TruthValue] = None,
        source_engine: Optional[str]        = None,
        metadata:      Optional[Dict[str, Any]] = None,
    ) -> Atom:
        """Add a link.  Merges if a matching link already exists."""
        if atom_type not in LINK_TYPES:
            raise ValueError(f"{atom_type} is not a link type")

        # Validate all targets exist
        for tid in outgoing_ids:
            if tid not in self._atoms:
                raise KeyError(f"Target atom {tid} not found in AtomSpace")

        tv = truth_value or TruthValue.DEFAULT()

        # Write gate
        threshold = compute_write_gate_threshold(
            self._eff_min_confidence_to_add,
            self.cor_level, self.da_level, self.cb1_level,
        )
        if tv.confidence < threshold:
            tv = TruthValue(strength=tv.strength, confidence=threshold)

        # Check for existing link with same type + outgoing
        existing = self._find_existing_link(atom_type, outgoing_ids)
        if existing is not None:
            merged_tv = merge_truth_values(existing.truth_value, tv)
            existing.truth_value = merged_tv
            if metadata:
                existing.metadata.update(metadata)
            self._atoms_merged += 1
            return existing

        # Capacity check
        if len(self._atoms) >= self.config.max_atoms:
            removed = self.enforce_capacity()
            if len(self._atoms) >= self.config.max_atoms and removed == 0:
                raise RuntimeError("AtomSpace at capacity; cannot add atom")

        atom = make_atom(
            atom_type=atom_type,
            name=None,
            outgoing=tuple(outgoing_ids),
            tv=tv,
            av=AttentionValue(),
            timetag=self._next_timetag,
            source_engine=source_engine,
            metadata=metadata,
        )
        self._next_timetag += 1

        # Store + index
        self._atoms[atom.atom_id] = atom
        self._by_type[atom_type].add(atom.atom_id)
        key = frozenset(outgoing_ids)
        self._by_outgoing[key].add(atom.atom_id)

        # Directional link indices
        if len(outgoing_ids) >= 1:
            self._links_from[(outgoing_ids[0], atom_type)].add(atom.atom_id)
        if len(outgoing_ids) >= 2:
            self._links_to[(outgoing_ids[1], atom_type)].add(atom.atom_id)

        # Update incoming sets on targets
        for tid in outgoing_ids:
            target = self._atoms.get(tid)
            if target is not None:
                target.incoming.add(atom.atom_id)

        self._novel_atoms_added += 1
        return atom

    def remove_atom(self, atom_id: str, cascade: bool = True) -> bool:
        """Remove an atom.  If *cascade*, also remove links that reference it."""
        atom = self._atoms.get(atom_id)
        if atom is None:
            return False

        # Cascade: remove all links referencing this atom
        if cascade and atom.incoming:
            incoming_copy = set(atom.incoming)
            for link_id in incoming_copy:
                self.remove_atom(link_id, cascade=True)

        # Remove from outgoing targets' incoming sets
        for tid in atom.outgoing:
            target = self._atoms.get(tid)
            if target is not None:
                target.incoming.discard(atom_id)

        # Remove from indices
        self._by_type.get(atom.atom_type, set()).discard(atom_id)
        if atom.name is not None:
            self._by_name.get(atom.name, set()).discard(atom_id)
        if atom.outgoing:
            key = frozenset(atom.outgoing)
            self._by_outgoing.get(key, set()).discard(atom_id)
            # Directional link indices
            if len(atom.outgoing) >= 1:
                self._links_from.get(
                    (atom.outgoing[0], atom.atom_type), set()
                ).discard(atom_id)
            if len(atom.outgoing) >= 2:
                self._links_to.get(
                    (atom.outgoing[1], atom.atom_type), set()
                ).discard(atom_id)

        del self._atoms[atom_id]
        return True

    def update_truth_value(self, atom_id: str, new_tv: TruthValue) -> None:
        """Replace truth value on an existing atom."""
        atom = self._atoms.get(atom_id)
        if atom is None:
            raise KeyError(f"Atom {atom_id} not found")
        atom.truth_value = new_tv

    def update_attention_value(self, atom_id: str, new_av: AttentionValue) -> None:
        """Replace attention value (called by ECAN)."""
        atom = self._atoms.get(atom_id)
        if atom is None:
            raise KeyError(f"Atom {atom_id} not found")
        atom.attention_value = new_av

    # -----------------------------------------------------------------
    # Read Operations
    # -----------------------------------------------------------------

    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """O(1) lookup by atom_id."""
        return self._atoms.get(atom_id)

    def get_by_name(self, name: str) -> List[Atom]:
        """Return all atoms with the given name."""
        ids = self._by_name.get(name, set())
        return [self._atoms[aid] for aid in ids if aid in self._atoms]

    def get_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Return all atoms of the given type."""
        ids = self._by_type.get(atom_type, set())
        return [self._atoms[aid] for aid in ids if aid in self._atoms]

    def get_incoming(self, atom_id: str) -> List[Atom]:
        """Return all links that reference this atom."""
        atom = self._atoms.get(atom_id)
        if atom is None:
            return []
        return [self._atoms[lid] for lid in atom.incoming if lid in self._atoms]

    def get_outgoing(self, atom_id: str) -> List[Atom]:
        """Return the atoms that this link points to."""
        atom = self._atoms.get(atom_id)
        if atom is None:
            return []
        return [self._atoms[tid] for tid in atom.outgoing if tid in self._atoms]

    def get_links_from(self, source_id: str, link_type: AtomType) -> List[Atom]:
        """O(k) lookup: all links of *link_type* whose outgoing[0] == source_id."""
        ids = self._links_from.get((source_id, link_type), set())
        return [self._atoms[aid] for aid in ids if aid in self._atoms]

    def get_links_to(self, target_id: str, link_type: AtomType) -> List[Atom]:
        """O(k) lookup: all links of *link_type* whose outgoing[1] == target_id."""
        ids = self._links_to.get((target_id, link_type), set())
        return [self._atoms[aid] for aid in ids if aid in self._atoms]

    def get_atoms_in_focus(self, threshold: float = 0.0) -> List[Atom]:
        """Return atoms with STI above *threshold* (Attentional Focus query)."""
        return [a for a in self._atoms.values() if a.attention_value.sti > threshold]

    def atom_count(self) -> int:
        """Total number of atoms."""
        return len(self._atoms)

    def get_all_atoms(self) -> List[Atom]:
        """Return all atoms in the space."""
        return list(self._atoms.values())

    # -----------------------------------------------------------------
    # Pattern Matching
    # -----------------------------------------------------------------

    def pattern_match(self, pattern: Pattern) -> List[Dict[str, Atom]]:
        """
        Match *pattern* against all links of ``pattern.root_type``.
        Returns list of variable binding dicts.
        """
        candidates = self.get_by_type(pattern.root_type)

        # NT modulation on scope
        max_results = self.config.max_match_results
        max_results = int(max_results * (1.0 + _W_NE * self.ne_level))

        # NT modulation on precision (raise effective TV thresholds)
        eff_tv_min_c = pattern.tv_min_confidence * (1.0 + _W_ACH * self.ach_level)

        adjusted_pattern = Pattern(
            root_type=pattern.root_type,
            elements=pattern.elements,
            tv_min_strength=pattern.tv_min_strength,
            tv_min_confidence=eff_tv_min_c,
        )

        results: List[Dict[str, Atom]] = []
        for link in candidates:
            if len(results) >= max_results:
                break
            bindings = match_pattern(adjusted_pattern, link, self._atoms)
            if bindings is not None:
                # Resolve atom_ids to Atoms in output
                resolved: Dict[str, Atom] = {}
                for var, aid in bindings.items():
                    atom = self._atoms.get(aid)
                    if atom is not None:
                        resolved[var] = atom
                if resolved or not bindings:
                    # Include the matched link itself
                    resolved["__link__"] = link
                    results.append(resolved)

        if results:
            self._match_results += len(results)
        else:
            self._match_failures += 1

        return results

    # -----------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------

    def decay_truth_values(self, dt: float = 1.0) -> int:
        """Decay confidence on all atoms.  Returns count of pruned atoms."""
        effective_rate = self.config.tv_decay_rate * (1.0 - _W_5HT * self._5ht_level)
        effective_rate = max(effective_rate, 0.0)

        prune_ids: List[str] = []
        for atom in list(self._atoms.values()):
            # Identity-relevant atoms don't decay
            if self.config.tv_identity_immune and atom.metadata.get("identity_relevant"):
                continue

            new_tv = compute_tv_decay(atom.truth_value, effective_rate * dt, self.config.tv_decay_floor)
            atom.truth_value = new_tv

            # Check pruning threshold
            if new_tv.confidence <= self._eff_prune_threshold_tv:
                # GABA accelerates pruning
                gaba_factor = 1.0 + _W_GABA * self.gaba_level
                if gaba_factor > 1.2 or new_tv.confidence <= self.config.tv_decay_floor:
                    if atom.attention_value.lti < self.config.prune_protect_lti:
                        prune_ids.append(atom.atom_id)

        for aid in prune_ids:
            self.remove_atom(aid, cascade=True)

        self._atoms_pruned += len(prune_ids)
        return len(prune_ids)

    def enforce_capacity(self) -> int:
        """Prune lowest-scored atoms to stay under max_atoms."""
        if len(self._atoms) < int(self.config.max_atoms * 0.9):
            return 0

        target = int(self.config.max_atoms * 0.8)
        candidates = [
            (score_atom_for_pruning(a), a.atom_id)
            for a in self._atoms.values()
            if not a.metadata.get("identity_relevant")
            and a.attention_value.lti < self.config.prune_protect_lti
        ]
        candidates.sort(key=lambda x: x[0])

        removed = 0
        for _score, aid in candidates:
            if len(self._atoms) - removed <= target:
                break
            if aid in self._atoms:
                self.remove_atom(aid, cascade=True)
                removed += 1

        self._atoms_pruned += removed
        return removed

    # -----------------------------------------------------------------
    # Memory Bridge
    # -----------------------------------------------------------------

    def export_to_dict(self, atom_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Serialize selected atoms (or all) to a dict for LTMM bridging.
        """
        ids = atom_ids or list(self._atoms.keys())
        exported: List[Dict[str, Any]] = []
        for aid in ids:
            atom = self._atoms.get(aid)
            if atom is None:
                continue
            exported.append({
                "atom_id":    atom.atom_id,
                "atom_type":  atom.atom_type.value,
                "name":       atom.name,
                "outgoing":   list(atom.outgoing),
                "tv_strength":   atom.truth_value.strength,
                "tv_confidence": atom.truth_value.confidence,
                "av_sti":     atom.attention_value.sti,
                "av_lti":     atom.attention_value.lti,
                "timetag":    atom.timetag,
                "source_engine": atom.source_engine,
                "metadata":   atom.metadata,
            })
        return {"atoms": exported, "next_timetag": self._next_timetag}

    def import_from_dict(self, data: Dict[str, Any]) -> List[str]:
        """
        Reconstruct atoms from a serialized dict.  Returns created atom_ids.
        """
        created: List[str] = []
        atom_map: Dict[str, str] = {}  # old_id → new_id (for link resolution)

        # Pass 1: Nodes
        for entry in data.get("atoms", []):
            atype = AtomType(entry["atom_type"])
            if is_node_type(atype):
                atom = Atom(
                    atom_id=entry["atom_id"],
                    atom_type=atype,
                    name=entry.get("name"),
                    outgoing=(),
                    truth_value=TruthValue(
                        strength=entry.get("tv_strength", 1.0),
                        confidence=entry.get("tv_confidence", 0.0),
                    ),
                    attention_value=AttentionValue(
                        sti=entry.get("av_sti", 0.0),
                        lti=entry.get("av_lti", 0.0),
                    ),
                    incoming=set(),
                    timetag=entry.get("timetag", self._next_timetag),
                    source_engine=entry.get("source_engine"),
                    metadata=entry.get("metadata", {}),
                )
                self._atoms[atom.atom_id] = atom
                self._by_type[atype].add(atom.atom_id)
                if atom.name is not None:
                    self._by_name[atom.name].add(atom.atom_id)
                atom_map[entry["atom_id"]] = atom.atom_id
                created.append(atom.atom_id)
                self._next_timetag = max(self._next_timetag, atom.timetag + 1)

        # Pass 2: Links
        for entry in data.get("atoms", []):
            atype = AtomType(entry["atom_type"])
            if is_link_type(atype):
                outgoing = tuple(
                    atom_map.get(oid, oid) for oid in entry.get("outgoing", [])
                )
                # Validate targets exist
                valid = all(oid in self._atoms for oid in outgoing)
                if not valid:
                    continue

                atom = Atom(
                    atom_id=entry["atom_id"],
                    atom_type=atype,
                    name=None,
                    outgoing=outgoing,
                    truth_value=TruthValue(
                        strength=entry.get("tv_strength", 1.0),
                        confidence=entry.get("tv_confidence", 0.0),
                    ),
                    attention_value=AttentionValue(
                        sti=entry.get("av_sti", 0.0),
                        lti=entry.get("av_lti", 0.0),
                    ),
                    incoming=set(),
                    timetag=entry.get("timetag", self._next_timetag),
                    source_engine=entry.get("source_engine"),
                    metadata=entry.get("metadata", {}),
                )
                self._atoms[atom.atom_id] = atom
                self._by_type[atype].add(atom.atom_id)
                key = frozenset(outgoing)
                self._by_outgoing[key].add(atom.atom_id)
                # Directional link indices
                if len(outgoing) >= 1:
                    self._links_from[(outgoing[0], atype)].add(atom.atom_id)
                if len(outgoing) >= 2:
                    self._links_to[(outgoing[1], atype)].add(atom.atom_id)
                for tid in outgoing:
                    target = self._atoms.get(tid)
                    if target is not None:
                        target.incoming.add(atom.atom_id)
                atom_map[entry["atom_id"]] = atom.atom_id
                created.append(atom.atom_id)
                self._next_timetag = max(self._next_timetag, atom.timetag + 1)

        return created

    # -----------------------------------------------------------------
    # process() — Main Entry Point
    # -----------------------------------------------------------------

    def process(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Pipeline-compatible entry point.

        input_data keys:
          "nt_state"  — Dict[str, float]
          "mode"      — str
          "commands"  — List[Dict] with keys: action, params

        Returns dict with results, stats, neurochem_signals.
        """
        input_data = input_data or {}

        # Reset cycle counters
        self._novel_atoms_added = 0
        self._atoms_merged = 0
        self._match_results = 0
        self._match_failures = 0
        self._atoms_pruned = 0

        # 1. NT state
        if "nt_state" in input_data:
            self.update_neurochem_state(input_data["nt_state"])

        # 2. Mode
        if "mode" in input_data:
            self._apply_mode_config(input_data["mode"])

        # 3. Execute commands
        results: List[Dict[str, Any]] = []
        for cmd in input_data.get("commands", []):
            result = self._execute_command(cmd)
            results.append(result)

        # 4. Maintenance
        if self._tick_counter > 0 and self._tick_counter % self.config.maintenance_interval == 0:
            self.decay_truth_values()
            self.enforce_capacity()

        self._tick_counter += 1

        # 5. Neurochem output
        signals = compute_atomspace_neurochem(
            self._novel_atoms_added,
            self._atoms_merged,
            self._match_results,
            self._match_failures,
            self._atoms_pruned,
        )

        return {
            "results":           results,
            "stats":             self._get_stats(),
            "neurochem_signals": signals.as_dict(),
        }

    def _execute_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a single command."""
        action = cmd.get("action", "")
        params = cmd.get("params", {})

        if action == "add_node":
            atom = self.add_node(
                atom_type=AtomType(params["atom_type"]),
                name=params["name"],
                truth_value=TruthValue(
                    strength=params.get("strength", 1.0),
                    confidence=params.get("confidence", 0.5),
                ),
                source_engine=params.get("source_engine"),
                metadata=params.get("metadata"),
            )
            return {"status": "ok", "atom_id": atom.atom_id}

        elif action == "add_link":
            atom = self.add_link(
                atom_type=AtomType(params["atom_type"]),
                outgoing_ids=tuple(params["outgoing_ids"]),
                truth_value=TruthValue(
                    strength=params.get("strength", 1.0),
                    confidence=params.get("confidence", 0.5),
                ),
                source_engine=params.get("source_engine"),
                metadata=params.get("metadata"),
            )
            return {"status": "ok", "atom_id": atom.atom_id}

        elif action == "remove":
            ok = self.remove_atom(
                params["atom_id"],
                cascade=params.get("cascade", True),
            )
            return {"status": "ok" if ok else "not_found"}

        elif action == "get":
            atom = self.get_atom(params["atom_id"])
            if atom is None:
                return {"status": "not_found"}
            return {
                "status": "ok",
                "atom_id": atom.atom_id,
                "atom_type": atom.atom_type.value,
                "name": atom.name,
                "tv_strength": atom.truth_value.strength,
                "tv_confidence": atom.truth_value.confidence,
            }

        elif action == "query_by_name":
            atoms = self.get_by_name(params["name"])
            return {
                "status": "ok",
                "atom_ids": [a.atom_id for a in atoms],
            }

        elif action == "query_by_type":
            atoms = self.get_by_type(AtomType(params["atom_type"]))
            return {
                "status": "ok",
                "atom_ids": [a.atom_id for a in atoms],
            }

        elif action == "pattern_match":
            elements = tuple(
                PatternAtom(
                    variable=e.get("variable"),
                    atom_type=AtomType(e["atom_type"]) if "atom_type" in e else None,
                    name=e.get("name"),
                    atom_id=e.get("atom_id"),
                )
                for e in params.get("elements", [])
            )
            pat = Pattern(
                root_type=AtomType(params["root_type"]),
                elements=elements,
                tv_min_strength=params.get("tv_min_strength", 0.0),
                tv_min_confidence=params.get("tv_min_confidence", 0.0),
            )
            matches = self.pattern_match(pat)
            serialized = []
            for binding in matches:
                entry: Dict[str, Any] = {}
                for var, atom in binding.items():
                    entry[var] = {"atom_id": atom.atom_id, "name": atom.name}
                serialized.append(entry)
            return {"status": "ok", "matches": serialized}

        elif action == "decay":
            count = self.decay_truth_values(dt=params.get("dt", 1.0))
            return {"status": "ok", "pruned": count}

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def _get_stats(self) -> Dict[str, Any]:
        """AtomSpace statistics."""
        type_counts: Dict[str, int] = {}
        for atype, ids in self._by_type.items():
            type_counts[atype.value] = len(ids)
        return {
            "total_atoms":  len(self._atoms),
            "type_counts":  type_counts,
            "next_timetag": self._next_timetag,
            "mode":         self._mode,
        }

    def get_status(self) -> Dict[str, Any]:
        """Standard engine introspection dict."""
        return {
            "engine_id":     self.engine_id,
            "cluster":       self.cluster,
            "total_atoms":   len(self._atoms),
            "next_timetag":  self._next_timetag,
            "tick_counter":  self._tick_counter,
            "mode":          self._mode,
            "nt_levels": {
                "5ht": self._5ht_level,
                "da":  self.da_level,
                "ne":  self.ne_level,
                "ach": self.ach_level,
                "gaba": self.gaba_level,
                "cor": self.cor_level,
                "oxt": self.oxt_level,
                "cb1": self.cb1_level,
            },
        }

    # -----------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------

    def _find_existing_node(
        self, atom_type: AtomType, name: str,
    ) -> Optional[Atom]:
        """Find an existing node with matching type and name."""
        candidates = self._by_name.get(name, set())
        for aid in candidates:
            atom = self._atoms.get(aid)
            if atom is not None and atom.atom_type == atom_type:
                return atom
        return None

    def _find_existing_link(
        self, atom_type: AtomType, outgoing_ids: Tuple[str, ...],
    ) -> Optional[Atom]:
        """Find an existing link with matching type and outgoing set.

        The ``_by_outgoing`` index uses ``frozenset`` keys (order-blind)
        for all link types, so the index lookup returns a superset of
        candidates for directed links.  The secondary check below
        enforces ordering for directed links and set-equality for
        symmetric links.
        """
        # Index lookup is always frozenset (matches add_link indexing)
        key = frozenset(outgoing_ids)

        candidates = self._by_outgoing.get(key, set())
        for aid in candidates:
            atom = self._atoms.get(aid)
            if atom is None or atom.atom_type != atom_type:
                continue
            # For directed links, order matters
            if atom_type not in SYMMETRIC_LINKS:
                if atom.outgoing == tuple(outgoing_ids):
                    return atom
            else:
                # Symmetric links — any permutation matches
                if set(atom.outgoing) == set(outgoing_ids):
                    return atom
        return None

    # -----------------------------------------------------------------
    # Persistence (CognitoolsDataStore)
    # -----------------------------------------------------------------

    def persist_to_store(self, store: Any) -> bool:
        """Persist current AtomSpace state to a CognitoolsDataStore.

        Parameters
        ----------
        store : CognitoolsDataStore
            KV store keyed by engine_id.

        Returns
        -------
        bool
            True if persisted successfully.
        """
        if store is None:
            return False
        try:
            data = self.export_to_dict()
            store.write("E9_atomspace", data)
            return True
        except Exception:
            return False

    def restore_from_store(self, store: Any) -> bool:
        """Restore AtomSpace state from a CognitoolsDataStore.

        Parameters
        ----------
        store : CognitoolsDataStore

        Returns
        -------
        bool
            True if restored successfully.
        """
        if store is None:
            return False
        try:
            data = store.get_by_id("E9_atomspace")
            if data is None:
                return False
            # Restore atoms from persisted data
            atoms_data = data.get("atoms", [])
            for atom_data in atoms_data:
                atom_type_str = atom_data.get("type", "")
                name = atom_data.get("name", "")
                if not atom_type_str:
                    continue
                try:
                    atype = AtomType(atom_type_str)
                except ValueError:
                    continue
                if atype in NODE_TYPES:
                    self.add_node(atype, name)
            return True
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self._atoms)

    def __repr__(self) -> str:
        return f"AtomSpaceEngine(atoms={len(self._atoms)}, timetag={self._next_timetag})"
