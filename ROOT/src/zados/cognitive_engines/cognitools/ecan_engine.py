"""
Engine 16 — ECAN Core  (Economic Attention Networks)
=====================================================

An attention economy for the AtomSpace (E9).  Atoms "compete" for
attention (STI), pay "rent" each cycle, earn "wages" when accessed,
and spread activation through HebbianLinks.  The result is a
self-organising priority system where important / relevant atoms
naturally rise to the top.

Key mechanics
-------------
- **STI** (Short-Term Importance): volatile attention currency
- **LTI** (Long-Term Importance): accumulated importance across sessions
- **Attentional Focus (AF)**: atoms with STI above threshold
- **HebbianLinks**: co-activation links strengthened by joint AF membership
- **Rent/Wage**: economic pressure that removes stale atoms

Spec: ``docs/specs/engine_16_ecan_spec.md``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.cognitools.atomspace_engine import (
    AtomSpaceEngine,
    AtomType,
    AttentionValue,
    TruthValue,
)

# =========================================================================
# 1. Configuration
# =========================================================================

@dataclass
class ECANConfig:
    # STI dynamics
    rent_rate:      float = 1.0
    wage_amount:    float = 10.0
    spread_rate:    float = 0.3
    sti_floor:      float = -200.0
    sti_ceiling:    float = 200.0

    # AF threshold
    af_threshold:   float = 10.0
    max_af_size:    int   = 100

    # HebbianLink dynamics
    hebbian_strengthen_rate:  float = 0.1
    hebbian_decay_rate:       float = 0.01
    hebbian_creation_threshold: int = 3    # Co-activations before creating link
    hebbian_min_strength:     float = 0.01 # Below this → remove HebbianLink

    # LTI dynamics
    lti_increment:      float = 0.1
    lti_decay_rate:     float = 0.01

    # Mode overrides
    mode_configs: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "ANALYTICAL": {
            "af_threshold":  15.0,
            "rent_rate":     1.5,
        },
        "CREATIVE": {
            "af_threshold":  5.0,
            "wage_amount":   15.0,
        },
        "REM_DREAM": {
            "af_threshold":  2.0,
            "rent_rate":     0.5,
            "spread_rate":   0.5,
        },
        "DEFAULT": {},
    })


# =========================================================================
# 2. Neurochemical Output
# =========================================================================

@dataclass
class ECANNeurochem:
    da_delta:        float = 0.0   # Atoms entering AF → DA (novelty)
    ne_delta:        float = 0.0   # Large AF → NE (arousal)
    ach_delta:       float = 0.0   # Focused AF → ACh
    _5ht_delta:      float = 0.0   # Hebbian consolidation → 5-HT
    gamma_boost:     float = 0.0   # Active spreading → gamma
    beta_boost:      float = 0.0   # Focused AF → beta

    def as_dict(self) -> Dict[str, float]:
        return {
            "da_delta":    self.da_delta,
            "ne_delta":    self.ne_delta,
            "ach_delta":   self.ach_delta,
            "_5ht_delta":  self._5ht_delta,
            "gamma_boost": self.gamma_boost,
            "beta_boost":  self.beta_boost,
        }


# =========================================================================
# 3. NT Modulation Weights
# =========================================================================

_W_NE   = 0.3   # Spreading rate
_W_ACH  = 0.3   # AF threshold tightening
_W_DA   = 0.3   # Wage boost
_W_GABA = 0.4   # Rent increase
_W_5HT  = 0.4   # Hebbian decay stabilization
_W_CB1  = 0.3   # AF threshold lowering
_W_OXT  = 0.2   # Social atom bonus


# =========================================================================
# 4. Pure Helper Functions
# =========================================================================

def compute_effective_rent(
    base: float,
    gaba: float,
) -> float:
    """NT-modulated rent rate."""
    return base * (1.0 + _W_GABA * gaba)


def compute_effective_wage(
    base: float,
    da: float,
) -> float:
    """NT-modulated wage amount."""
    return base * (1.0 + _W_DA * da)


def compute_effective_spread_rate(
    base: float,
    ne: float,
) -> float:
    """NT-modulated spreading rate."""
    return base * (1.0 + _W_NE * ne)


def compute_effective_af_threshold(
    base: float,
    ach: float,
    cb1: float,
) -> float:
    """NT-modulated AF threshold."""
    return base * (1.0 + _W_ACH * ach) * max(1.0 - _W_CB1 * cb1, 0.3)


def compute_effective_hebbian_decay(
    base: float,
    sht: float,
) -> float:
    """NT-modulated Hebbian decay rate. 5-HT stabilizes."""
    return base * max(1.0 - _W_5HT * sht, 0.1)


def compute_ecan_neurochem(
    atoms_entered_af: int,
    af_size: int,
    hebbian_strengthened: int,
    spread_events: int,
) -> ECANNeurochem:
    """Compute NT output from ECAN cycle events."""
    nc = ECANNeurochem()
    nc.da_delta    = atoms_entered_af * 0.05
    nc.ne_delta    = min(af_size * 0.01, 0.3)
    nc.ach_delta   = 0.05 if 0 < af_size <= 10 else 0.0
    nc._5ht_delta  = hebbian_strengthened * 0.03
    nc.gamma_boost = spread_events * 0.01
    nc.beta_boost  = 0.05 if 0 < af_size <= 20 else 0.0
    return nc


# =========================================================================
# 5. ECAN Engine
# =========================================================================

class ECANEngine:
    """
    Engine 16 — Economic Attention Networks.

    Manages an attention economy over AtomSpace atoms.  Each cycle:
    rent is charged, wages are paid, activation spreads through
    HebbianLinks, and the Attentional Focus is recomputed.
    """

    engine_id: str = "ecan_engine"
    cluster:   str = "knowledge_substrate"

    def __init__(
        self,
        atomspace: AtomSpaceEngine,
        config: Optional[ECANConfig] = None,
    ) -> None:
        self.atomspace = atomspace
        self.config = config or ECANConfig()

        # NT levels (Pattern A)
        self.ne_level:   float = 0.5
        self.ach_level:  float = 0.5
        self.da_level:   float = 0.5
        self.gaba_level: float = 0.5
        self._5ht_level: float = 0.5
        self.cb1_level:  float = 0.5
        self.oxt_level:  float = 0.5

        # Effective params
        self._eff_rent:       float = self.config.rent_rate
        self._eff_wage:       float = self.config.wage_amount
        self._eff_spread:     float = self.config.spread_rate
        self._eff_af_thresh:  float = self.config.af_threshold
        self._eff_heb_decay:  float = self.config.hebbian_decay_rate

        # State
        self._mode:           str           = "DEFAULT"
        self._tick_counter:   int           = 0
        self._accessed_atoms: Set[str]      = set()   # Atoms accessed this cycle
        self._prev_af:        Set[str]      = set()   # AF from previous cycle
        self._coactivation_counts: Dict[Tuple[str, str], int] = {}

        # Cycle counters
        self._atoms_entered_af:      int = 0
        self._hebbian_strengthened:  int = 0
        self._spread_events:         int = 0

    # -----------------------------------------------------------------
    # Pattern A: Neurochemical State Update
    # -----------------------------------------------------------------

    def update_neurochem_state(self, nt_state: Dict[str, float]) -> None:
        self.ne_level   = _clamp(nt_state.get("ne",   self.ne_level))
        self.ach_level  = _clamp(nt_state.get("ach",  self.ach_level))
        self.da_level   = _clamp(nt_state.get("da",   self.da_level))
        self.gaba_level = _clamp(nt_state.get("gaba", self.gaba_level))
        self._5ht_level = _clamp(nt_state.get("5ht",  self._5ht_level))
        self.cb1_level  = _clamp(nt_state.get("cb1",  self.cb1_level))
        self.oxt_level  = _clamp(nt_state.get("oxt",  self.oxt_level))
        self._recompute_effective_params()

    def _recompute_effective_params(self) -> None:
        mode_overrides = self.config.mode_configs.get(self._mode, {})
        base_rent = mode_overrides.get("rent_rate", self.config.rent_rate)
        base_wage = mode_overrides.get("wage_amount", self.config.wage_amount)
        base_spread = mode_overrides.get("spread_rate", self.config.spread_rate)
        base_af = mode_overrides.get("af_threshold", self.config.af_threshold)

        self._eff_rent   = compute_effective_rent(base_rent, self.gaba_level)
        self._eff_wage   = compute_effective_wage(base_wage, self.da_level)
        self._eff_spread = compute_effective_spread_rate(base_spread, self.ne_level)
        self._eff_af_thresh = compute_effective_af_threshold(base_af, self.ach_level, self.cb1_level)
        self._eff_heb_decay = compute_effective_hebbian_decay(
            self.config.hebbian_decay_rate, self._5ht_level,
        )

    # -----------------------------------------------------------------
    # Mode Configuration
    # -----------------------------------------------------------------

    def _apply_mode_config(self, mode: str) -> None:
        self._mode = mode
        self._recompute_effective_params()

    # -----------------------------------------------------------------
    # Core: Mark atoms as accessed
    # -----------------------------------------------------------------

    def mark_accessed(self, atom_id: str) -> None:
        """Mark an atom as accessed this cycle (earns wage)."""
        self._accessed_atoms.add(atom_id)

    # -----------------------------------------------------------------
    # Core: Run one ECAN cycle
    # -----------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """
        Execute one ECAN cycle: rent → spread → wage → clamp → AF → Hebbian.
        """
        # Reset cycle counters
        self._atoms_entered_af = 0
        self._hebbian_strengthened = 0
        self._spread_events = 0

        all_atoms = self.atomspace.get_all_atoms()

        # --- 1. RENT ---
        for atom in all_atoms:
            atom.attention_value.sti -= self._eff_rent

        # --- 2. SPREAD activation through HebbianLinks ---
        hebbian_links = self.atomspace.get_by_type(AtomType.HEBBIAN_LINK)
        spread_deltas: Dict[str, float] = {}

        for heb in hebbian_links:
            if len(heb.outgoing) != 2:
                continue
            a_id, b_id = heb.outgoing
            atom_a = self.atomspace.get_atom(a_id)
            atom_b = self.atomspace.get_atom(b_id)
            if atom_a is None or atom_b is None:
                continue

            heb_strength = heb.truth_value.strength

            # A → B spreading
            if atom_a.attention_value.sti > 0:
                delta = self._eff_spread * atom_a.attention_value.sti * heb_strength
                spread_deltas[b_id] = spread_deltas.get(b_id, 0.0) + delta
                self._spread_events += 1

            # B → A spreading (symmetric)
            if atom_b.attention_value.sti > 0:
                delta = self._eff_spread * atom_b.attention_value.sti * heb_strength
                spread_deltas[a_id] = spread_deltas.get(a_id, 0.0) + delta
                self._spread_events += 1

        # Apply spread deltas
        for atom_id, delta in spread_deltas.items():
            atom = self.atomspace.get_atom(atom_id)
            if atom is not None:
                atom.attention_value.sti += delta

        # --- 3. WAGE for accessed atoms ---
        for atom_id in self._accessed_atoms:
            atom = self.atomspace.get_atom(atom_id)
            if atom is not None:
                bonus = self._eff_wage
                # OXT bonus for social-context atoms
                if atom.metadata.get("social_context"):
                    bonus += _W_OXT * self.oxt_level * 10.0
                atom.attention_value.sti += bonus

        # --- 4. CLAMP STI ---
        for atom in all_atoms:
            atom.attention_value.sti = max(
                self.config.sti_floor,
                min(self.config.sti_ceiling, atom.attention_value.sti),
            )

        # --- 5. Compute Attentional Focus ---
        current_af: Set[str] = set()
        af_atoms = sorted(
            [a for a in self.atomspace.get_all_atoms()
             if a.attention_value.sti > self._eff_af_thresh],
            key=lambda a: a.attention_value.sti,
            reverse=True,
        )
        for atom in af_atoms[:self.config.max_af_size]:
            current_af.add(atom.atom_id)

        # Track atoms entering AF
        new_in_af = current_af - self._prev_af
        self._atoms_entered_af = len(new_in_af)

        # --- 6. LTI updates ---
        for atom in self.atomspace.get_all_atoms():
            if atom.atom_id in current_af:
                atom.attention_value.lti += self.config.lti_increment
            else:
                atom.attention_value.lti = max(
                    0.0, atom.attention_value.lti - self.config.lti_decay_rate,
                )

        # --- 7. HebbianLink management ---
        self._update_hebbian_links(current_af)

        # --- 8. Hebbian decay ---
        self._decay_hebbian_links()

        # Save AF for next cycle
        self._prev_af = current_af
        self._accessed_atoms.clear()
        self._tick_counter += 1

        # Neurochem output
        signals = compute_ecan_neurochem(
            self._atoms_entered_af,
            len(current_af),
            self._hebbian_strengthened,
            self._spread_events,
        )

        return {
            "af_size":      len(current_af),
            "af_atom_ids":  list(current_af),
            "atoms_entered_af": self._atoms_entered_af,
            "spread_events": self._spread_events,
            "neurochem_signals": signals.as_dict(),
        }

    # -----------------------------------------------------------------
    # HebbianLink Management
    # -----------------------------------------------------------------

    def _update_hebbian_links(self, current_af: Set[str]) -> None:
        """Strengthen/create HebbianLinks for co-activated atoms."""
        af_list = list(current_af)
        for i in range(len(af_list)):
            for j in range(i + 1, len(af_list)):
                a_id, b_id = af_list[i], af_list[j]
                pair = (min(a_id, b_id), max(a_id, b_id))

                # Check if HebbianLink already exists
                existing = self._find_hebbian_link(a_id, b_id)
                if existing is not None:
                    # Strengthen
                    new_s = min(
                        existing.truth_value.strength + self.config.hebbian_strengthen_rate,
                        1.0,
                    )
                    self.atomspace.update_truth_value(
                        existing.atom_id,
                        TruthValue(strength=new_s, confidence=existing.truth_value.confidence),
                    )
                    self._hebbian_strengthened += 1
                else:
                    # Track co-activation count
                    count = self._coactivation_counts.get(pair, 0) + 1
                    self._coactivation_counts[pair] = count

                    if count >= self.config.hebbian_creation_threshold:
                        self.atomspace.add_link(
                            AtomType.HEBBIAN_LINK,
                            (a_id, b_id),
                            TruthValue(strength=0.1, confidence=0.5),
                            source_engine=self.engine_id,
                        )
                        del self._coactivation_counts[pair]
                        self._hebbian_strengthened += 1

    def _decay_hebbian_links(self) -> None:
        """Decay HebbianLink strengths. Remove weak ones."""
        hebbian_links = self.atomspace.get_by_type(AtomType.HEBBIAN_LINK)
        to_remove: List[str] = []

        for heb in hebbian_links:
            new_s = heb.truth_value.strength - self._eff_heb_decay
            if new_s <= self.config.hebbian_min_strength:
                to_remove.append(heb.atom_id)
            else:
                self.atomspace.update_truth_value(
                    heb.atom_id,
                    TruthValue(strength=new_s, confidence=heb.truth_value.confidence),
                )

        for aid in to_remove:
            self.atomspace.remove_atom(aid)

    def _find_hebbian_link(self, a_id: str, b_id: str) -> Any:
        """Find existing HebbianLink between two atoms."""
        hebbian_links = self.atomspace.get_by_type(AtomType.HEBBIAN_LINK)
        for heb in hebbian_links:
            if len(heb.outgoing) == 2:
                if set(heb.outgoing) == {a_id, b_id}:
                    return heb
        return None

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    def get_attentional_focus(self) -> List[str]:
        """Return atom_ids in the Attentional Focus."""
        atoms = self.atomspace.get_all_atoms()
        af = sorted(
            [a for a in atoms if a.attention_value.sti > self._eff_af_thresh],
            key=lambda a: a.attention_value.sti,
            reverse=True,
        )
        return [a.atom_id for a in af[:self.config.max_af_size]]

    def get_af_size(self) -> int:
        """Return current AF size."""
        return len(self.get_attentional_focus())

    # -----------------------------------------------------------------
    # process() — Pipeline Entry Point
    # -----------------------------------------------------------------

    def process(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        input_data = input_data or {}

        # 1. NT state
        if "nt_state" in input_data:
            self.update_neurochem_state(input_data["nt_state"])

        # 2. Mode
        if "mode" in input_data:
            self._apply_mode_config(input_data["mode"])

        # 3. Mark accessed atoms
        for atom_id in input_data.get("accessed_atoms", []):
            self.mark_accessed(atom_id)

        # 4. Run one ECAN cycle
        result = self.step()

        return result

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id":      self.engine_id,
            "cluster":        self.cluster,
            "mode":           self._mode,
            "tick_counter":   self._tick_counter,
            "af_size":        self.get_af_size(),
            "eff_rent":       self._eff_rent,
            "eff_wage":       self._eff_wage,
            "eff_spread":     self._eff_spread,
            "eff_af_thresh":  self._eff_af_thresh,
            "nt_levels": {
                "ne":   self.ne_level,
                "ach":  self.ach_level,
                "da":   self.da_level,
                "gaba": self.gaba_level,
                "5ht":  self._5ht_level,
                "cb1":  self.cb1_level,
                "oxt":  self.oxt_level,
            },
        }

    def __repr__(self) -> str:
        return f"ECANEngine(mode={self._mode}, af_size={self.get_af_size()})"
