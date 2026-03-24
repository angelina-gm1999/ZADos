"""
Batch simulation runner with parameter sweeps (Appendix N.4/N.7).

Orchestrates full simulation runs using the NeurochemicalEngine with
all 12 NTs, optional scheduler, logger, and checkpointing.

Usage
-----
>>> from zados.neurochem.optimization.batch_runner import (
...     SimulationConfig, run_simulation, run_batch,
...     generate_parameter_grid, generate_lhs_configs,
... )
>>> config = SimulationConfig(n_steps=1000, seed=42)
>>> result = run_simulation(config)
>>> print(result.elapsed_steps)
"""

from __future__ import annotations

import copy
import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np

from zados.neurochem.optimization.timescale import (
    TimescaleConfig,
    SparseUpdateScheduler,
)
from zados.neurochem.optimization.logging import (
    LogTierConfig,
    HierarchicalLogger,
)
from zados.neurochem.optimization.checkpoint import (
    Checkpoint,
    create_checkpoint,
)


@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuration for a single simulation run.

    Parameters
    ----------
    dt : float
        Integration time step.
    n_steps : int
        Number of simulation steps.
    seed : int
        RNG seed for reproducibility.
    scheduler : TimescaleConfig, optional
        Sparse update scheduling config. None = every-tick updates.
    log_tiers : list of LogTierConfig, optional
        Logging tier configs. None = no logging.
    checkpoint_interval : int
        Steps between checkpoints. 0 = no checkpointing.
    oscillation_mode : str
        Oscillation update mode ("static" or "state_derived").
    parameter_overrides : dict, optional
        {nt_name: {param: value}} overrides applied to DEFAULT_NT_CONFIGS.
    signal_fn : callable, optional
        Function(step_number) → modulation_signals dict.
        Called each step to provide input signals.
    """
    dt: float = 0.01
    n_steps: int = 1000
    seed: int = 42
    scheduler: Optional[TimescaleConfig] = None
    log_tiers: Optional[Tuple[LogTierConfig, ...]] = None
    checkpoint_interval: int = 0
    oscillation_mode: str = "state_derived"
    parameter_overrides: Optional[Dict[str, Dict[str, float]]] = None
    signal_fn: Optional[Callable] = None


@dataclass
class SimulationResult:
    """
    Result of a completed simulation run.

    Parameters
    ----------
    config : SimulationConfig
        The config used for this run.
    logger : HierarchicalLogger, optional
        Logger with recorded data (if logging was enabled).
    final_checkpoint : Checkpoint, optional
        Final state checkpoint.
    elapsed_steps : int
        Number of steps completed.
    """
    config: SimulationConfig
    logger: Optional[HierarchicalLogger] = None
    final_checkpoint: Optional[Checkpoint] = None
    elapsed_steps: int = 0


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """
    Execute a single simulation run (N.4).

    Creates engine, registers all NTs/receptors/modules, attaches
    scheduler + logger, runs step loop, returns result.

    Parameters
    ----------
    config : SimulationConfig

    Returns
    -------
    SimulationResult
    """
    from zados.neurochem.core.engine import NeurochemicalEngine
    from zados.neurochem.neurotransmitters.configs import register_all_neurotransmitters
    from zados.neurochem.neurotransmitters.module_registry import register_all_nt_modules
    from zados.neurochem.neurotransmitters.configs import register_all_receptor_modules_on_engine

    # Create engine
    engine = NeurochemicalEngine(
        dt=config.dt,
        seed=config.seed,
        oscillation_mode=config.oscillation_mode,
    )

    # Register all NT systems
    register_all_neurotransmitters(engine)

    # Apply parameter overrides
    if config.parameter_overrides:
        for nt_name, overrides in config.parameter_overrides.items():
            try:
                cfg = engine.registry.get_config(nt_name)
                cfg.update(overrides)
            except KeyError:
                pass

    # Register modules
    register_all_nt_modules(engine)
    register_all_receptor_modules_on_engine(engine)

    # Attach scheduler
    if config.scheduler is not None:
        engine.scheduler = SparseUpdateScheduler(config.scheduler)

    # Set up logger
    logger = None
    if config.log_tiers is not None:
        logger = HierarchicalLogger(list(config.log_tiers), max_steps=config.n_steps)

    # Run simulation loop
    last_checkpoint = None
    for step in range(config.n_steps):
        # Compute signals
        signals = None
        if config.signal_fn is not None:
            signals = config.signal_fn(step)

        engine.step(signals)

        # Logging
        if logger is not None:
            for tier_name, tier in logger.tiers.items():
                if logger.should_log(tier_name, step):
                    for var in tier.variables:
                        if var == "concentrations":
                            logger.log_concentrations(step, engine)
                        elif var == "receptors":
                            logger.log_receptors(step, engine)
                        elif var == "oscillations":
                            logger.log_oscillations(step, engine)

        # Checkpointing
        if config.checkpoint_interval > 0 and (step + 1) % config.checkpoint_interval == 0:
            last_checkpoint = create_checkpoint(engine, step_number=step + 1)

    # Final checkpoint
    if last_checkpoint is None or last_checkpoint.step_number != config.n_steps:
        last_checkpoint = create_checkpoint(engine, step_number=config.n_steps)

    return SimulationResult(
        config=config,
        logger=logger,
        final_checkpoint=last_checkpoint,
        elapsed_steps=config.n_steps,
    )


def generate_parameter_grid(
    base_config: SimulationConfig,
    sweep_params: Dict[str, List[float]],
) -> List[SimulationConfig]:
    """
    Generate grid of configs from parameter sweep (N.7.1).

    Parameters
    ----------
    base_config : SimulationConfig
        Base configuration to clone.
    sweep_params : dict
        {param_path: [values]} where param_path is "NT_NAME.param_name"
        e.g., {"DA.C_baseline": [0.3, 0.4, 0.5], "5HT.theta_tonic": [0.05, 0.1]}

    Returns
    -------
    list of SimulationConfig
        One config per grid point.
    """
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[k] for k in param_names]

    configs = []
    for i, combo in enumerate(itertools.product(*param_values)):
        overrides = dict(base_config.parameter_overrides or {})
        for name, val in zip(param_names, combo):
            nt_name, param_name = name.split(".", 1)
            if nt_name not in overrides:
                overrides[nt_name] = {}
            overrides[nt_name][param_name] = val

        configs.append(SimulationConfig(
            dt=base_config.dt,
            n_steps=base_config.n_steps,
            seed=base_config.seed + i,
            scheduler=base_config.scheduler,
            log_tiers=base_config.log_tiers,
            checkpoint_interval=base_config.checkpoint_interval,
            oscillation_mode=base_config.oscillation_mode,
            parameter_overrides=overrides,
            signal_fn=base_config.signal_fn,
        ))

    return configs


def generate_lhs_configs(
    base_config: SimulationConfig,
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int,
    seed: int = 0,
) -> List[SimulationConfig]:
    """
    Generate Latin Hypercube Sampled configs (N.7.2).

    Parameters
    ----------
    base_config : SimulationConfig
        Base configuration.
    param_ranges : dict
        {param_path: (min_val, max_val)} where param_path is "NT_NAME.param_name"
    n_samples : int
        Number of samples.
    seed : int
        RNG seed for LHS.

    Returns
    -------
    list of SimulationConfig
    """
    rng = np.random.default_rng(seed)
    param_names = list(param_ranges.keys())
    n_dims = len(param_names)

    # Latin Hypercube: one sample per interval per dimension
    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples[i, d] = (perm[i] + rng.uniform()) / n_samples

    configs = []
    for i in range(n_samples):
        overrides = dict(base_config.parameter_overrides or {})
        for d, name in enumerate(param_names):
            nt_name, param_name = name.split(".", 1)
            lo, hi = param_ranges[name]
            val = lo + samples[i, d] * (hi - lo)
            if nt_name not in overrides:
                overrides[nt_name] = {}
            overrides[nt_name][param_name] = val

        configs.append(SimulationConfig(
            dt=base_config.dt,
            n_steps=base_config.n_steps,
            seed=base_config.seed + i,
            scheduler=base_config.scheduler,
            log_tiers=base_config.log_tiers,
            checkpoint_interval=base_config.checkpoint_interval,
            oscillation_mode=base_config.oscillation_mode,
            parameter_overrides=overrides,
            signal_fn=base_config.signal_fn,
        ))

    return configs


def run_batch(
    configs: List[SimulationConfig],
    max_workers: Optional[int] = None,
) -> List[SimulationResult]:
    """
    Run batch of simulations (N.7.3).

    Uses concurrent.futures.ProcessPoolExecutor for parallelism.
    Falls back to sequential execution if max_workers=1.

    Parameters
    ----------
    configs : list of SimulationConfig
        Simulation configurations to run.
    max_workers : int, optional
        Maximum parallel workers. None = auto (cpu_count).

    Returns
    -------
    list of SimulationResult
        Results in same order as configs.
    """
    if max_workers == 1 or len(configs) <= 1:
        return [run_simulation(c) for c in configs]

    # For multiprocessing, signal_fn must be None (not picklable)
    for c in configs:
        if c.signal_fn is not None:
            raise ValueError(
                "signal_fn is not supported in parallel batch mode "
                "(not picklable). Use max_workers=1 for sequential execution "
                "or remove signal_fn."
            )

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(run_simulation, configs))

    return results
