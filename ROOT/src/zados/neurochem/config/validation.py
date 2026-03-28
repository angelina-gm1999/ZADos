"""
Configuration validation utilities.

Provides validation functions for NT and receptor configs,
ensuring all required keys exist and values are within
biologically-reasonable bounds.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple


# Required keys for NT configs
NT_REQUIRED_KEYS = [
    "C_baseline", "theta_tonic", "theta_phasic",
    "sigma_tonic", "sigma_phasic",
    "u_base", "d_base", "c_base",
]

# Required keys for receptor configs
RECEPTOR_REQUIRED_KEYS = [
    "K_d", "parent_nt", "exposure_tau",
]

# Reasonable bounds for NT config values
NT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "C_baseline": (0.0, 1.0),
    "theta_tonic": (0.0, 2.0),
    "theta_phasic": (0.0, 5.0),
    "sigma_tonic": (0.0, 0.5),
    "sigma_phasic": (0.0, 0.5),
    "u_base": (0.0, 1.0),
    "d_base": (0.0, 1.0),
    "c_base": (0.0, 1.0),
}

# Reasonable bounds for receptor config values
RECEPTOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "K_d": (0.01, 2.0),
    "exposure_tau": (1.0, 100.0),
}


def validate_nt_config(
    config: Dict[str, Any],
    nt_name: str = "",
) -> List[str]:
    """
    Validate a neurotransmitter configuration dict.

    Parameters
    ----------
    config : dict
        NT config dict to validate
    nt_name : str, optional
        NT name for error messages

    Returns
    -------
    list of str
        List of validation error messages (empty = valid)
    """
    errors: List[str] = []
    prefix = f"[{nt_name}] " if nt_name else ""

    # Check required keys
    for key in NT_REQUIRED_KEYS:
        if key not in config:
            errors.append(f"{prefix}Missing required key: {key}")

    # Check value types and bounds
    for key, (lo, hi) in NT_BOUNDS.items():
        if key in config:
            val = config[key]
            if not isinstance(val, (int, float)):
                errors.append(f"{prefix}{key}: expected numeric, got {type(val).__name__}")
            elif val < lo or val > hi:
                errors.append(
                    f"{prefix}{key}={val} out of bounds [{lo}, {hi}]"
                )

    # Sanity checks
    if "sigma_tonic" in config and "sigma_phasic" in config:
        if config["sigma_phasic"] < config["sigma_tonic"]:
            errors.append(
                f"{prefix}sigma_phasic ({config['sigma_phasic']}) < "
                f"sigma_tonic ({config['sigma_tonic']})"
            )

    return errors


def validate_receptor_config(
    config: Dict[str, Any],
    receptor_id: str = "",
) -> List[str]:
    """
    Validate a receptor configuration dict.

    Parameters
    ----------
    config : dict
        Receptor config dict to validate
    receptor_id : str, optional
        Receptor ID for error messages

    Returns
    -------
    list of str
        List of validation error messages (empty = valid)
    """
    errors: List[str] = []
    prefix = f"[{receptor_id}] " if receptor_id else ""

    # Check required keys
    for key in RECEPTOR_REQUIRED_KEYS:
        if key not in config:
            errors.append(f"{prefix}Missing required key: {key}")

    # Check value types and bounds
    for key, (lo, hi) in RECEPTOR_BOUNDS.items():
        if key in config:
            val = config[key]
            if not isinstance(val, (int, float)):
                errors.append(f"{prefix}{key}: expected numeric, got {type(val).__name__}")
            elif val < lo or val > hi:
                errors.append(
                    f"{prefix}{key}={val} out of bounds [{lo}, {hi}]"
                )

    # parent_nt must be a non-empty string
    if "parent_nt" in config:
        if not isinstance(config["parent_nt"], str) or not config["parent_nt"]:
            errors.append(f"{prefix}parent_nt must be a non-empty string")

    return errors


def validate_all_configs(
    nt_configs: Dict[str, Dict[str, Any]],
    receptor_configs: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    Validate all NT and receptor configs.

    Returns
    -------
    list of str
        Combined list of all validation errors
    """
    errors = []
    for nt_name, config in nt_configs.items():
        errors.extend(validate_nt_config(config, nt_name))
    for receptor_id, config in receptor_configs.items():
        errors.extend(validate_receptor_config(config, receptor_id))
    return errors
