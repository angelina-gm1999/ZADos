"""
State checkpointing and resume (Appendix N.8).

Serializes/restores the complete NeurochemicalEngine state including
all NT states, receptor states, oscillation state, configs, RNG, and time.

Uses existing as_dict()/from_dict() on NeurotransmitterState, ReceptorState,
and OscillationState for serialization.

Usage
-----
>>> from zados.neurochem.optimization.checkpoint import (
...     create_checkpoint, restore_checkpoint,
...     checkpoint_to_dict, checkpoint_from_dict,
... )
>>> ckpt = create_checkpoint(engine, step_number=500)
>>> data = checkpoint_to_dict(ckpt)  # JSON-safe dict
>>> ckpt2 = checkpoint_from_dict(data)
>>> restore_checkpoint(engine, ckpt2)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from zados.neurochem.optimization.seeding import (
    save_single_rng_state,
    restore_single_rng_state,
)


@dataclass(frozen=True)
class Checkpoint:
    """
    Complete engine state snapshot (N.8.1).

    All sub-dicts use plain Python types (JSON-safe) via as_dict().
    """
    step_number: int
    current_time: float
    neurotransmitter_states: Dict[str, dict] = field(default_factory=dict)
    receptor_states: Dict[str, dict] = field(default_factory=dict)
    oscillation_state: Optional[dict] = None
    configs: Dict[str, dict] = field(default_factory=dict)
    rng_state: Optional[dict] = None
    effective_signaling: Dict[str, float] = field(default_factory=dict)


def create_checkpoint(engine: Any, step_number: int) -> Checkpoint:
    """
    Snapshot current engine state.

    Parameters
    ----------
    engine : NeurochemicalEngine
        Engine to snapshot.
    step_number : int
        Current step number.

    Returns
    -------
    Checkpoint
        Frozen snapshot of all engine state.
    """
    # NT states
    nt_states = {}
    for nt_name, state in engine.registry.iter_neurotransmitters():
        nt_states[nt_name] = state.as_dict()

    # Receptor states
    receptor_states = {}
    for receptor_id, state in engine.registry.iter_receptors():
        receptor_states[receptor_id] = state.as_dict()

    # Oscillation state
    osc = engine.registry.get_oscillations()
    osc_dict = osc.as_dict() if osc is not None else None

    # Configs (deep copy to avoid mutation)
    configs = {}
    for nt_name in engine.registry.neurotransmitter_names():
        try:
            configs[nt_name] = copy.deepcopy(engine.registry.get_config(nt_name))
        except KeyError:
            pass
    for receptor_id in engine.registry.receptor_ids():
        try:
            configs[receptor_id] = copy.deepcopy(engine.registry.get_config(receptor_id))
        except KeyError:
            pass

    # RNG state
    rng_state = None
    if hasattr(engine, 'rng') and engine.rng is not None:
        rng_state = save_single_rng_state(engine.rng)

    # Effective signaling
    effective_signaling = engine.registry.get_all_effective_signaling()

    return Checkpoint(
        step_number=step_number,
        current_time=engine.current_time,
        neurotransmitter_states=nt_states,
        receptor_states=receptor_states,
        oscillation_state=osc_dict,
        configs=configs,
        rng_state=rng_state,
        effective_signaling=effective_signaling,
    )


def restore_checkpoint(engine: Any, checkpoint: Checkpoint) -> None:
    """
    Restore engine to checkpointed state.

    Mutates the engine in-place.

    Parameters
    ----------
    engine : NeurochemicalEngine
        Engine to restore.
    checkpoint : Checkpoint
        State to restore from.
    """
    from zados.neurochem.state import (
        NeurotransmitterState,
        ReceptorState,
        OscillationState,
    )

    # Restore time
    engine.current_time = checkpoint.current_time
    engine._step_number = checkpoint.step_number

    # Restore NT states
    for nt_name, state_dict in checkpoint.neurotransmitter_states.items():
        restored = NeurotransmitterState.from_dict(state_dict)
        engine.registry._neurotransmitters[nt_name] = restored

    # Restore receptor states
    for receptor_id, state_dict in checkpoint.receptor_states.items():
        restored = ReceptorState.from_dict(state_dict)
        engine.registry._receptors[receptor_id] = restored

    # Restore oscillation state
    if checkpoint.oscillation_state is not None:
        restored_osc = OscillationState.from_dict(checkpoint.oscillation_state)
        engine.registry.set_oscillations(restored_osc)

    # Restore configs
    for name, config in checkpoint.configs.items():
        engine.registry._configs[name] = copy.deepcopy(config)

    # Restore RNG
    if checkpoint.rng_state is not None and hasattr(engine, 'rng'):
        restore_single_rng_state(engine.rng, checkpoint.rng_state)

    # Restore effective signaling
    engine.registry._effective_signaling = dict(checkpoint.effective_signaling)


def checkpoint_to_dict(checkpoint: Checkpoint) -> dict:
    """
    Serialize checkpoint to plain dict (JSON-safe).

    Parameters
    ----------
    checkpoint : Checkpoint

    Returns
    -------
    dict
    """
    return {
        "step_number": checkpoint.step_number,
        "current_time": checkpoint.current_time,
        "neurotransmitter_states": checkpoint.neurotransmitter_states,
        "receptor_states": checkpoint.receptor_states,
        "oscillation_state": checkpoint.oscillation_state,
        "configs": checkpoint.configs,
        "rng_state": _serialize_rng_state(checkpoint.rng_state),
        "effective_signaling": checkpoint.effective_signaling,
    }


def checkpoint_from_dict(data: dict) -> Checkpoint:
    """
    Deserialize checkpoint from dict.

    Parameters
    ----------
    data : dict

    Returns
    -------
    Checkpoint
    """
    return Checkpoint(
        step_number=data["step_number"],
        current_time=data["current_time"],
        neurotransmitter_states=data.get("neurotransmitter_states", {}),
        receptor_states=data.get("receptor_states", {}),
        oscillation_state=data.get("oscillation_state"),
        configs=data.get("configs", {}),
        rng_state=_deserialize_rng_state(data.get("rng_state")),
        effective_signaling=data.get("effective_signaling", {}),
    )


def _serialize_rng_state(state: Optional[dict]) -> Optional[dict]:
    """Convert numpy RNG state to JSON-safe format."""
    if state is None:
        return None
    # Deep copy and convert numpy arrays to lists
    result = copy.deepcopy(state)
    if "state" in result and "state" in result["state"]:
        s = result["state"]["state"]
        if hasattr(s, "tolist"):
            result["state"]["state"] = s.tolist()
    return result


def _deserialize_rng_state(data: Optional[dict]) -> Optional[dict]:
    """Convert JSON-safe RNG state back to numpy format."""
    if data is None:
        return None
    import numpy as np
    result = copy.deepcopy(data)
    if "state" in result and "state" in result["state"]:
        s = result["state"]["state"]
        if isinstance(s, list):
            result["state"]["state"] = np.array(s, dtype=np.uint64)
    return result
