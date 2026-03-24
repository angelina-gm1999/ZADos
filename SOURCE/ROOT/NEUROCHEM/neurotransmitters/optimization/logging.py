"""
Hierarchical multi-resolution logging (Appendix N.6).

Samples different variable groups at different rates and stores them
in pre-allocated numpy arrays. Saves to .npz files.

Usage
-----
>>> from zados.neurochem.optimization.logging import (
...     HierarchicalLogger, LogTierConfig, DEFAULT_LOG_TIERS,
... )
>>> logger = HierarchicalLogger(DEFAULT_LOG_TIERS, max_steps=10000)
>>> for step in range(10000):
...     engine.step()
...     if logger.should_log("high_res", step):
...         logger.log_concentrations(step, engine)
>>> logger.save_npz("run_001.npz")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class LogTierConfig:
    """
    Configuration for one logging tier.

    Parameters
    ----------
    name : str
        Tier identifier (e.g., "high_res", "med_res").
    sample_interval : int
        Log every N steps.
    variables : list of str
        What to log: "concentrations", "receptors", "oscillations", "metrics".
    """
    name: str
    sample_interval: int
    variables: List[str] = field(default_factory=list)


DEFAULT_LOG_TIERS: List[LogTierConfig] = [
    LogTierConfig("high_res", 10, ["concentrations"]),
    LogTierConfig("med_res", 100, ["receptors", "oscillations"]),
    LogTierConfig("low_res", 100, ["metrics"]),
]


class HierarchicalLogger:
    """
    Multi-resolution time series logger (N.6.1).

    Pre-allocates numpy arrays for each tier and grows dynamically
    if the initial allocation is exceeded.
    """

    def __init__(
        self,
        tiers: Optional[List[LogTierConfig]] = None,
        max_steps: int = 100000,
    ):
        if tiers is None:
            tiers = DEFAULT_LOG_TIERS

        self.tiers: Dict[str, LogTierConfig] = {t.name: t for t in tiers}
        self.max_steps = max_steps

        # Storage: {tier_name: {variable_name: list of arrays}}
        # We use lists-of-dicts for flexibility, then convert to arrays at save time
        self._data: Dict[str, List[dict]] = {t.name: [] for t in tiers}
        self._steps: Dict[str, List[int]] = {t.name: [] for t in tiers}

    def should_log(self, tier_name: str, step_number: int) -> bool:
        """Check if tier should sample on this step."""
        tier = self.tiers.get(tier_name)
        if tier is None:
            return False
        if tier.sample_interval <= 1:
            return True
        return step_number % tier.sample_interval == 0

    def log_concentrations(self, step: int, engine: Any) -> None:
        """
        Record all NT concentrations (C_tonic, C_phasic, F).

        Stores as dict {nt_name_component: float} per step.
        """
        record = {}
        for nt_name, state in engine.registry.iter_neurotransmitters():
            record[f"{nt_name}_C_tonic"] = state.C_tonic
            record[f"{nt_name}_C_phasic"] = state.C_phasic
            record[f"{nt_name}_F"] = state.F
        # Find tier that has "concentrations"
        for tier_name, tier in self.tiers.items():
            if "concentrations" in tier.variables:
                self._data[tier_name].append(record)
                self._steps[tier_name].append(step)
                break

    def log_receptors(self, step: int, engine: Any) -> None:
        """
        Record receptor states (rho, sigma) for all receptors.
        """
        record = {}
        for receptor_id, state in engine.registry.iter_receptors():
            record[f"{receptor_id}_rho"] = state.rho
            record[f"{receptor_id}_sigma"] = state.sigma
        for tier_name, tier in self.tiers.items():
            if "receptors" in tier.variables:
                self._data[tier_name].append(record)
                self._steps[tier_name].append(step)
                break

    def log_oscillations(self, step: int, engine: Any) -> None:
        """
        Record oscillation band amplitudes.
        """
        osc = engine.registry.get_oscillations()
        if osc is None:
            return
        record = {
            "delta": osc.delta,
            "theta": osc.theta,
            "alpha": osc.alpha,
            "beta": osc.beta,
            "gamma": osc.gamma,
        }
        for tier_name, tier in self.tiers.items():
            if "oscillations" in tier.variables:
                self._data[tier_name].append(record)
                self._steps[tier_name].append(step)
                break

    def log_metrics(self, step: int, metrics: dict) -> None:
        """
        Record neurosymbolic metrics.
        """
        record = dict(metrics)
        for tier_name, tier in self.tiers.items():
            if "metrics" in tier.variables:
                self._data[tier_name].append(record)
                self._steps[tier_name].append(step)
                break

    def get_tier_data(self, tier_name: str) -> Dict[str, np.ndarray]:
        """
        Get arrays for a specific tier.

        Returns
        -------
        dict
            {variable_name: np.ndarray of shape (n_logged_steps,)},
            plus "steps" key with step indices.
        """
        records = self._data.get(tier_name, [])
        steps = self._steps.get(tier_name, [])
        if not records:
            return {"steps": np.array([], dtype=np.int64)}

        # Collect all keys from first record
        keys = list(records[0].keys())
        result = {"steps": np.array(steps, dtype=np.int64)}
        for key in keys:
            result[key] = np.array([r.get(key, 0.0) for r in records], dtype=np.float32)
        return result

    def save_npz(self, path: str) -> None:
        """
        Save all tiers to .npz file.

        Keys in the .npz are formatted as "{tier_name}__{variable_name}".
        """
        arrays = {}
        for tier_name in self.tiers:
            tier_data = self.get_tier_data(tier_name)
            for var_name, arr in tier_data.items():
                arrays[f"{tier_name}__{var_name}"] = arr

        # Save tier metadata
        tier_names = list(self.tiers.keys())
        arrays["__tier_names__"] = np.array(tier_names, dtype=object)

        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: str) -> "HierarchicalLogger":
        """
        Load from .npz file.

        Returns a logger with pre-populated data (read-only, no tier configs).
        """
        data = np.load(path, allow_pickle=True)
        tier_names = list(data.get("__tier_names__", []))

        # Reconstruct tiers with dummy configs
        tiers = [LogTierConfig(name=name, sample_interval=1) for name in tier_names]
        logger = cls(tiers=tiers, max_steps=0)

        # Populate data from arrays
        for key in data.files:
            if key == "__tier_names__":
                continue
            if "__" not in key:
                continue
            tier_name, var_name = key.split("__", 1)
            if tier_name not in logger.tiers:
                continue

            arr = data[key]
            if var_name == "steps":
                logger._steps[tier_name] = arr.tolist()
            else:
                # Reconstruct records from column arrays
                n = len(arr)
                while len(logger._data[tier_name]) < n:
                    logger._data[tier_name].append({})
                for i in range(n):
                    logger._data[tier_name][i][var_name] = float(arr[i])

        return logger
