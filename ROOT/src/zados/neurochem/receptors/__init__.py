"""
Per-receptor-family behavior modules.

Each module defines pharmacodynamic specifications for a receptor family:
- ReceptorSpec per subtype (ionotropic/metabotropic, signaling type)
- Effective signaling proxy computation (A_ij)
- Emotion-layer plasticity hooks
"""

from .base import ReceptorFamilyModule, ReceptorSpec
from .receptor_registry import ReceptorModuleRegistry, register_all_receptor_modules
from .plasticity import compute_plasticity_deltas, apply_plasticity_delta
from .subtype_switching import (
    SubtypeSwitchRule,
    compute_subtype_switch_deltas,
    apply_subtype_switch_deltas,
)

__all__ = [
    "ReceptorFamilyModule",
    "ReceptorSpec",
    "ReceptorModuleRegistry",
    "register_all_receptor_modules",
    "compute_plasticity_deltas",
    "apply_plasticity_delta",
    "SubtypeSwitchRule",
    "compute_subtype_switch_deltas",
    "apply_subtype_switch_deltas",
]
