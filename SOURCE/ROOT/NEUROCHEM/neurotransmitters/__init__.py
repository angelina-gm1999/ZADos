from .dopamine import Dopamine, DAModule
from .configs import (
    DEFAULT_NT_CONFIGS,
    DEFAULT_RECEPTOR_CONFIGS,
    NT_RECEPTOR_MAP,
    register_neurotransmitter,
    register_all_neurotransmitters,
)
from .base import (
    NeurotransmitterModule,
    ReleaseDriveSpec,
    OscillationCouplingRule,
)
from .module_registry import NTModuleRegistry, register_all_nt_modules

__all__ = [
    # Legacy (batch simulation)
    "Dopamine",
    # Module system (online engine)
    "NeurotransmitterModule",
    "ReleaseDriveSpec",
    "OscillationCouplingRule",
    "DAModule",
    "NTModuleRegistry",
    "register_all_nt_modules",
    # Config data
    "DEFAULT_NT_CONFIGS",
    "DEFAULT_RECEPTOR_CONFIGS",
    "NT_RECEPTOR_MAP",
    "register_neurotransmitter",
    "register_all_neurotransmitters",
]
