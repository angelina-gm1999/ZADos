"""
Registry for receptor family modules.

Provides lookup and batch registration of all receptor family modules.
Parallels neurotransmitters/module_registry.py for the receptor layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from zados.neurochem.receptors.base import ReceptorFamilyModule


class ReceptorModuleRegistry:
    """
    Central registry for receptor family modules.

    Keyed by parent NT name (e.g., "DA", "5HT").
    """

    _modules: Dict[str, ReceptorFamilyModule] = {}

    @classmethod
    def register(cls, module: ReceptorFamilyModule) -> None:
        """Register a receptor family module."""
        cls._modules[module.parent_nt] = module

    @classmethod
    def get(cls, parent_nt: str) -> Optional[ReceptorFamilyModule]:
        """Get receptor family module by parent NT name."""
        return cls._modules.get(parent_nt)

    @classmethod
    def is_registered(cls, parent_nt: str) -> bool:
        """Check if a receptor family is registered."""
        return parent_nt in cls._modules

    @classmethod
    def registered_names(cls) -> List[str]:
        """Return sorted list of registered parent NT names."""
        return sorted(cls._modules.keys())

    @classmethod
    def count(cls) -> int:
        """Return number of registered receptor families."""
        return len(cls._modules)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered modules."""
        cls._modules.clear()

    @classmethod
    def get_all(cls) -> Dict[str, ReceptorFamilyModule]:
        """Return dict of all registered modules."""
        return dict(cls._modules)


def register_all_receptor_modules() -> None:
    """
    Register all 11 receptor family modules.

    Cortisol has no receptor dynamics (concentration-only),
    so it has no receptor family module.
    """
    from zados.neurochem.receptors.dopamine_receptors import DopamineReceptors
    from zados.neurochem.receptors.serotonin_receptors import SerotoninReceptors
    from zados.neurochem.receptors.norepinephrine_receptors import NorepinephrineReceptors
    from zados.neurochem.receptors.acetylcholine_receptors import AcetylcholineReceptors
    from zados.neurochem.receptors.oxytocin_receptors import OxytocinReceptors
    from zados.neurochem.receptors.opioid_receptors import OpioidReceptors
    from zados.neurochem.receptors.cannabinoid_receptors import CannabinoidReceptors
    from zados.neurochem.receptors.crh_receptors import CRHReceptors
    from zados.neurochem.receptors.gaba_receptors import GABAReceptors
    from zados.neurochem.receptors.glutamate_receptors import GlutamateReceptors
    from zados.neurochem.receptors.histamine_receptors import HistamineReceptors

    ALL_RECEPTOR_MODULES = [
        DopamineReceptors,
        SerotoninReceptors,
        NorepinephrineReceptors,
        AcetylcholineReceptors,
        OxytocinReceptors,
        OpioidReceptors,
        CannabinoidReceptors,
        CRHReceptors,
        GABAReceptors,
        GlutamateReceptors,
        HistamineReceptors,
    ]

    for module_cls in ALL_RECEPTOR_MODULES:
        ReceptorModuleRegistry.register(module_cls())
