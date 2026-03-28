"""
ZADOS Neurochemical Optimization Package (Appendix N).

Provides computational optimization strategies for offline batch simulation:
- Deterministic seeding and RNG management (N.5)
- State checkpointing and resume (N.8)
- Timescale separation with sparse update scheduling (N.2)
- Hierarchical multi-resolution logging (N.6)
- Batch simulation runner with parameter sweeps (N.4/N.7)
- Post-processing analysis utilities (N.9)
"""

from .seeding import (
    make_seed_sequence,
    derive_rng,
    create_rng_registry,
    save_rng_states,
    restore_rng_states,
)
from .checkpoint import (
    Checkpoint,
    create_checkpoint,
    restore_checkpoint,
    checkpoint_to_dict,
    checkpoint_from_dict,
)
from .timescale import (
    TimescaleConfig,
    SparseUpdateScheduler,
    DEFAULT_TIMESCALE_CONFIG,
)
from .logging import (
    LogTierConfig,
    HierarchicalLogger,
    DEFAULT_LOG_TIERS,
)
from .batch_runner import (
    SimulationConfig,
    SimulationResult,
    run_simulation,
    generate_parameter_grid,
    generate_lhs_configs,
    run_batch,
)
from .analysis import (
    temporal_mean,
    temporal_variance,
    temporal_std,
    cross_run_statistics,
)

__all__ = [
    # Seeding (N.5)
    "make_seed_sequence",
    "derive_rng",
    "create_rng_registry",
    "save_rng_states",
    "restore_rng_states",
    # Checkpointing (N.8)
    "Checkpoint",
    "create_checkpoint",
    "restore_checkpoint",
    "checkpoint_to_dict",
    "checkpoint_from_dict",
    # Timescale separation (N.2)
    "TimescaleConfig",
    "SparseUpdateScheduler",
    "DEFAULT_TIMESCALE_CONFIG",
    # Logging (N.6)
    "LogTierConfig",
    "HierarchicalLogger",
    "DEFAULT_LOG_TIERS",
    # Batch runner (N.4/N.7)
    "SimulationConfig",
    "SimulationResult",
    "run_simulation",
    "generate_parameter_grid",
    "generate_lhs_configs",
    "run_batch",
    # Analysis (N.9)
    "temporal_mean",
    "temporal_variance",
    "temporal_std",
    "cross_run_statistics",
]
