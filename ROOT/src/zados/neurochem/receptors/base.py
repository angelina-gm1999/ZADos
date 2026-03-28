"""
Base classes for per-receptor-family behavior modules.

Each receptor family gets a concrete subclass that defines:
- ReceptorSpec per subtype (binding type, signaling mode)
- Effective signaling proxy (A_ij = rho * sigma * gamma * g(chi) * S)
- Emotion-layer plasticity hooks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from zados.neurochem.oscillations.oscillation_modulation import compute_g_chi


@dataclass(frozen=True)
class ReceptorSpec:
    """
    Pharmacodynamic specification for a single receptor subtype.

    Attributes
    ----------
    receptor_id : str
        Matches configs.py key (e.g., "DA_D1", "5HT_2A")
    ionotropic : bool
        True for fast ligand-gated ion channels, False for metabotropic/GPCR
    signaling_type : str
        "excitatory" | "inhibitory" | "modulatory"
    effective_signaling_weight : float
        Base weight for the effective signaling proxy A_ij.
        Default 1.0 (no scaling).
    emotion_plasticity_rules : dict
        Maps emotion_id -> dict of receptor parameter deltas.
        e.g. {"joy": {"sigma_delta": 0.1}, "fear": {"sigma_delta": -0.2}}
    """
    receptor_id: str
    ionotropic: bool = False
    signaling_type: str = "modulatory"
    effective_signaling_weight: float = 1.0
    emotion_plasticity_rules: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        valid_types = {"excitatory", "inhibitory", "modulatory"}
        if self.signaling_type not in valid_types:
            raise ValueError(
                f"Invalid signaling_type {self.signaling_type!r}. "
                f"Must be one of {sorted(valid_types)}"
            )


class ReceptorFamilyModule(ABC):
    """
    Abstract base for receptor family behavior specification.

    One module per NT's receptor family (e.g., DopamineReceptors for DA_D1..D5).
    """

    @property
    @abstractmethod
    def parent_nt(self) -> str:
        """Parent neurotransmitter name (e.g., 'DA', '5HT')."""
        ...

    @property
    @abstractmethod
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        """Map of receptor_id -> ReceptorSpec for this family."""
        ...

    def compute_effective_signaling(
        self,
        receptor_id: str,
        rho: float,
        sigma: float,
        functional_state: str,
        saturation: float,
        gamma_gprotein: float = 1.0,
    ) -> float:
        """
        Compute effective signaling proxy A_ij per PDF Appendix D.

        A_ij = rho * sigma * gamma * g(chi) * S_ij * w_ij

        Parameters
        ----------
        receptor_id : str
            Receptor subtype identifier
        rho : float
            Receptor density in [0, 1]
        sigma : float
            Receptor sensitivity in [0, 1]
        functional_state : str
            One of "ACTIVE", "DESENSITIZED", "INTERNALIZED", "UPREGULATED"
        saturation : float
            Ligand binding saturation S_ij in [0, 1]
        gamma_gprotein : float, default=1.0
            G-protein coupling efficacy in [0, 1]

        Returns
        -------
        float
            Effective signaling value (non-negative)
        """
        spec = self.receptor_specs.get(receptor_id)
        weight = spec.effective_signaling_weight if spec else 1.0
        g_chi = compute_g_chi(functional_state)
        return rho * sigma * gamma_gprotein * g_chi * saturation * weight

    def get_receptor_ids(self) -> List[str]:
        """Return sorted list of receptor IDs in this family."""
        return sorted(self.receptor_specs.keys())

    def get_emotion_plasticity(
        self,
        receptor_id: str,
        emotion_id: str,
    ) -> Optional[Dict[str, float]]:
        """
        Get emotion-specific plasticity rules for a receptor.

        Returns None if no rules defined for this emotion/receptor pair.
        """
        spec = self.receptor_specs.get(receptor_id)
        if spec is None:
            return None
        return spec.emotion_plasticity_rules.get(emotion_id)

    @property
    def subtype_switch_rules(self) -> list:
        """
        Subtype switching rules for this receptor family.

        Override in subclasses to define density transfer rules between
        receptor subtypes under sustained activation.

        Returns
        -------
        list
            List of SubtypeSwitchRule instances (empty by default)
        """
        return []
