"""
Typed configuration wrappers for neurochemical parameters.

Provides frozen dataclass wrappers over the dict-based configs
in neurotransmitters/configs.py, enabling IDE autocomplete,
type checking, and validation.
"""

from .nt_config import NTConfig
from .receptor_config import ReceptorConfig
from .validation import validate_nt_config, validate_receptor_config

__all__ = [
    "NTConfig",
    "ReceptorConfig",
    "validate_nt_config",
    "validate_receptor_config",
]
