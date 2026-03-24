"""
Post-processing analysis utilities (Appendix N.9).

Provides functions for computing statistics over logged simulation data
and across multiple simulation runs.

Usage
-----
>>> from zados.neurochem.optimization.analysis import (
...     temporal_mean, temporal_variance, cross_run_statistics,
... )
>>> mean_vals = temporal_mean(data_array)
>>> stats = cross_run_statistics(results, "conc", "DA_C_tonic")
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from zados.neurochem.optimization.batch_runner import SimulationResult


def temporal_mean(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute mean over the temporal (first) axis.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_steps, ...) or (n_steps,).
    axis : int
        Axis to reduce over. Default 0 (time).

    Returns
    -------
    np.ndarray
        Mean values with temporal axis collapsed.
    """
    return np.mean(data, axis=axis)


def temporal_variance(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute variance over the temporal (first) axis.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_steps, ...) or (n_steps,).
    axis : int
        Axis to reduce over. Default 0 (time).

    Returns
    -------
    np.ndarray
        Variance values with temporal axis collapsed.
    """
    return np.var(data, axis=axis)


def temporal_std(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute standard deviation over the temporal axis.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_steps, ...) or (n_steps,).
    axis : int
        Axis to reduce over. Default 0 (time).

    Returns
    -------
    np.ndarray
        Std values with temporal axis collapsed.
    """
    return np.std(data, axis=axis)


def cross_run_statistics(
    results: List[SimulationResult],
    tier_name: str,
    variable: str,
) -> Dict[str, np.ndarray]:
    """
    Compute cross-run statistics for a logged variable (N.9.1).

    Extracts the named variable from each run's logger at the given tier,
    stacks them, and computes mean/std/min/max across runs.

    Parameters
    ----------
    results : list of SimulationResult
        Completed simulation results (must have loggers with matching tiers).
    tier_name : str
        Logger tier name (e.g., "conc", "high_res").
    variable : str
        Variable key within the tier data (e.g., "DA_C_tonic").

    Returns
    -------
    dict
        {
            "mean": np.ndarray,   # mean across runs at each timestep
            "std": np.ndarray,    # std across runs
            "min": np.ndarray,    # min across runs
            "max": np.ndarray,    # max across runs
            "n_runs": int,        # number of runs
        }

    Raises
    ------
    ValueError
        If no results have loggers or the variable is missing.
    """
    traces = []
    for r in results:
        if r.logger is None:
            continue
        data = r.logger.get_tier_data(tier_name)
        if variable not in data:
            continue
        traces.append(np.asarray(data[variable]))

    if not traces:
        raise ValueError(
            f"No valid traces found for tier={tier_name!r}, variable={variable!r}. "
            f"Ensure results have loggers with the requested tier and variable."
        )

    # Trim to minimum length (in case runs have different step counts)
    min_len = min(len(t) for t in traces)
    stacked = np.stack([t[:min_len] for t in traces], axis=0)  # (n_runs, n_steps)

    return {
        "mean": np.mean(stacked, axis=0),
        "std": np.std(stacked, axis=0),
        "min": np.min(stacked, axis=0),
        "max": np.max(stacked, axis=0),
        "n_runs": len(traces),
    }
