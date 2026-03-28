"""
Deterministic seeding and RNG management (Appendix N.5).

Uses numpy's SeedSequence for hierarchical seed derivation,
providing reproducible, independent random streams per NT/receptor/run.

Usage
-----
>>> from zados.neurochem.optimization.seeding import (
...     make_seed_sequence, derive_rng, create_rng_registry,
...     save_rng_states, restore_rng_states,
... )
>>> ss = make_seed_sequence(42)
>>> rng = derive_rng(ss, "dopamine_noise", run_id=0)
>>> rng.normal(0, 1)  # reproducible draw
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def make_seed_sequence(global_seed: int) -> np.random.SeedSequence:
    """
    Create a root SeedSequence from a global seed.

    Parameters
    ----------
    global_seed : int
        Global seed for the entire simulation.

    Returns
    -------
    np.random.SeedSequence
        Root sequence that can spawn child sequences.
    """
    return np.random.SeedSequence(global_seed)


def derive_rng(
    seed_seq: np.random.SeedSequence,
    stream_name: str,
    run_id: int = 0,
) -> np.random.Generator:
    """
    Derive a per-stream RNG from a root SeedSequence (N.5.1).

    Uses SeedSequence.spawn() with entropy mixing from (run_id, stream_name)
    to produce independent, deterministic random streams.

    Parameters
    ----------
    seed_seq : np.random.SeedSequence
        Root or parent SeedSequence.
    stream_name : str
        Name of the random stream (e.g., "dopamine_noise").
    run_id : int, default=0
        Run index for batch execution.

    Returns
    -------
    np.random.Generator
        Independent RNG for this stream.
    """
    # Derive a child sequence with entropy from run_id and stream_name hash
    stream_hash = hash(stream_name) & 0xFFFFFFFF
    child = seed_seq.spawn(1)[0]
    # Further spawn with run_id and stream_name for unique derivation
    sub = np.random.SeedSequence(
        entropy=child.entropy,
        spawn_key=(run_id, stream_hash),
    )
    return np.random.default_rng(sub)


def create_rng_registry(
    seed_seq: np.random.SeedSequence,
    stream_names: List[str],
    run_id: int = 0,
) -> Dict[str, np.random.Generator]:
    """
    Create a dict of named RNG streams from one root SeedSequence.

    Parameters
    ----------
    seed_seq : np.random.SeedSequence
        Root sequence.
    stream_names : list of str
        Names for each stream (e.g., ["DA", "5HT", "NE"]).
    run_id : int, default=0
        Run index.

    Returns
    -------
    dict
        {stream_name: np.random.Generator}
    """
    return {
        name: derive_rng(seed_seq, name, run_id)
        for name in stream_names
    }


def save_rng_states(
    rng_registry: Dict[str, np.random.Generator],
) -> Dict[str, dict]:
    """
    Serialize all RNG states for checkpoint (N.5.2).

    Parameters
    ----------
    rng_registry : dict
        {name: np.random.Generator}

    Returns
    -------
    dict
        {name: bit_generator_state_dict}
    """
    return {
        name: rng.bit_generator.state
        for name, rng in rng_registry.items()
    }


def restore_rng_states(
    rng_registry: Dict[str, np.random.Generator],
    saved: Dict[str, dict],
) -> None:
    """
    Restore RNG states from checkpoint data (N.5.2).

    Mutates the generators in-place.

    Parameters
    ----------
    rng_registry : dict
        {name: np.random.Generator} — will be mutated.
    saved : dict
        {name: bit_generator_state_dict} from save_rng_states.
    """
    for name, state in saved.items():
        if name in rng_registry:
            rng_registry[name].bit_generator.state = state


def save_single_rng_state(rng: np.random.Generator) -> dict:
    """
    Serialize a single RNG's state.

    Parameters
    ----------
    rng : np.random.Generator

    Returns
    -------
    dict
        bit_generator state dict
    """
    return rng.bit_generator.state


def restore_single_rng_state(rng: np.random.Generator, state: dict) -> None:
    """
    Restore a single RNG's state.

    Parameters
    ----------
    rng : np.random.Generator
        Will be mutated.
    state : dict
        From save_single_rng_state.
    """
    rng.bit_generator.state = state
