"""
ZADOS Neurochemical Simulation Layer.

Provides real-time and batch neurochemical simulation with per-NT modules,
receptor dynamics, oscillation coupling, and neurosymbolic readout.
"""

from .core import NeurochemicalEngine, SimulationRunner, NeurochemicalRegistry
from .state import NeurotransmitterState, ReceptorState, OscillationState
from .neurotransmitters import (
    NeurotransmitterModule,
    ReleaseDriveSpec,
    OscillationCouplingRule,
    DAModule,
    NTModuleRegistry,
    register_all_nt_modules,
    register_all_neurotransmitters,
    DEFAULT_NT_CONFIGS,
    DEFAULT_RECEPTOR_CONFIGS,
    NT_RECEPTOR_MAP,
)
from .receptors import (
    ReceptorFamilyModule,
    ReceptorSpec,
    ReceptorModuleRegistry,
    register_all_receptor_modules,
)
from .config import NTConfig, ReceptorConfig, validate_nt_config, validate_receptor_config
from .neurosymbolic.readout import compute_neurosymbolic_readout
from .optimization import (
    # Seeding
    make_seed_sequence,
    derive_rng,
    # Checkpointing
    Checkpoint,
    create_checkpoint,
    restore_checkpoint,
    # Timescale
    TimescaleConfig,
    SparseUpdateScheduler,
    # Logging
    LogTierConfig,
    HierarchicalLogger,
    # Batch runner
    SimulationConfig,
    SimulationResult,
    run_simulation,
    run_batch,
    generate_parameter_grid,
    generate_lhs_configs,
    # Analysis
    temporal_mean,
    temporal_variance,
    cross_run_statistics,
)

__all__ = [
    # Core engine
    "NeurochemicalEngine",
    "SimulationRunner",
    "NeurochemicalRegistry",
    # State containers
    "NeurotransmitterState",
    "ReceptorState",
    "OscillationState",
    # NT module system
    "NeurotransmitterModule",
    "ReleaseDriveSpec",
    "OscillationCouplingRule",
    "DAModule",
    "NTModuleRegistry",
    "register_all_nt_modules",
    # NT registration + configs
    "register_all_neurotransmitters",
    "DEFAULT_NT_CONFIGS",
    "DEFAULT_RECEPTOR_CONFIGS",
    "NT_RECEPTOR_MAP",
    # Receptor module system
    "ReceptorFamilyModule",
    "ReceptorSpec",
    "ReceptorModuleRegistry",
    "register_all_receptor_modules",
    # Typed config wrappers
    "NTConfig",
    "ReceptorConfig",
    "validate_nt_config",
    "validate_receptor_config",
    # Neurosymbolic readout
    "compute_neurosymbolic_readout",
    # Optimization (Appendix N)
    "make_seed_sequence",
    "derive_rng",
    "Checkpoint",
    "create_checkpoint",
    "restore_checkpoint",
    "TimescaleConfig",
    "SparseUpdateScheduler",
    "LogTierConfig",
    "HierarchicalLogger",
    "SimulationConfig",
    "SimulationResult",
    "run_simulation",
    "run_batch",
    "generate_parameter_grid",
    "generate_lhs_configs",
    "temporal_mean",
    "temporal_variance",
    "cross_run_statistics",
]
