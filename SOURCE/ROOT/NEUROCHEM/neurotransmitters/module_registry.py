"""
Registry for per-NT behavior modules.

Provides lookup by NT name so the engine can dispatch to the correct
module during its update loop.

Usage
-----
>>> from zados.neurochem.neurotransmitters.module_registry import (
...     NTModuleRegistry, register_all_nt_modules,
... )
>>> register_all_nt_modules()
>>> da_module = NTModuleRegistry.get("DA")
>>> drive = da_module.compute_release_drive({"novelty": 0.8, "rpe": 0.3})
"""

from __future__ import annotations

from typing import Dict, List, Optional

from zados.neurochem.neurotransmitters.base import NeurotransmitterModule


class NTModuleRegistry:
    """
    Registry of per-NT behavior modules.

    Class-level registry that stores NeurotransmitterModule instances
    keyed by their NT name.
    """

    _modules: Dict[str, NeurotransmitterModule] = {}

    @classmethod
    def register(cls, module: NeurotransmitterModule) -> None:
        """
        Register a per-NT behavior module.

        Parameters
        ----------
        module : NeurotransmitterModule
            Module instance to register. Keyed by module.name.
        """
        cls._modules[module.name] = module

    @classmethod
    def get(cls, nt_name: str) -> Optional[NeurotransmitterModule]:
        """
        Look up a module by NT name.

        Parameters
        ----------
        nt_name : str
            NT identifier (e.g., "DA", "5HT")

        Returns
        -------
        NeurotransmitterModule or None
            Module if registered, None otherwise
        """
        return cls._modules.get(nt_name)

    @classmethod
    def all_modules(cls) -> Dict[str, NeurotransmitterModule]:
        """Return a copy of all registered modules."""
        return dict(cls._modules)

    @classmethod
    def registered_names(cls) -> List[str]:
        """Return sorted list of registered NT names."""
        return sorted(cls._modules.keys())

    @classmethod
    def is_registered(cls, nt_name: str) -> bool:
        """Check if a module is registered for the given NT name."""
        return nt_name in cls._modules

    @classmethod
    def clear(cls) -> None:
        """Clear all registered modules. Mainly for testing."""
        cls._modules = {}

    @classmethod
    def count(cls) -> int:
        """Return number of registered modules."""
        return len(cls._modules)


def register_all_nt_modules(engine=None) -> None:
    """
    Instantiate and register all 12 per-NT behavior modules.

    If an engine is provided, also registers the modules on the engine
    so it can dispatch to them during update.

    Parameters
    ----------
    engine : NeurochemicalEngine, optional
        If provided, calls engine.register_nt_module() for each module.
    """
    from zados.neurochem.neurotransmitters.dopamine import DAModule
    from zados.neurochem.neurotransmitters.serotonin import SerotoninModule
    from zados.neurochem.neurotransmitters.norepinephrine import NEModule
    from zados.neurochem.neurotransmitters.acetylcholine import AChModule
    from zados.neurochem.neurotransmitters.oxytocin import OXTModule
    from zados.neurochem.neurotransmitters.opioid import MORModule
    from zados.neurochem.neurotransmitters.endocannabinoid import CB1Module
    from zados.neurochem.neurotransmitters.cortisol_mod import CortisolModule
    from zados.neurochem.neurotransmitters.crh import CRHModule
    from zados.neurochem.neurotransmitters.gaba import GABAModule
    from zados.neurochem.neurotransmitters.glutamate import GLUModule
    from zados.neurochem.neurotransmitters.histamine_mod import HistamineModule

    modules = [
        DAModule(),
        SerotoninModule(),
        NEModule(),
        AChModule(),
        OXTModule(),
        MORModule(),
        CB1Module(),
        CortisolModule(),
        CRHModule(),
        GABAModule(),
        GLUModule(),
        HistamineModule(),
    ]

    for mod in modules:
        NTModuleRegistry.register(mod)
        if engine is not None:
            engine.register_nt_module(mod)
