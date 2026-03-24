"""Tests for state checkpointing (Appendix N.8)."""

import pytest

from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.neurotransmitters.configs import (
    register_all_neurotransmitters,
    DEFAULT_NT_CONFIGS,
    NT_RECEPTOR_MAP,
)
from zados.neurochem.optimization.checkpoint import (
    Checkpoint,
    create_checkpoint,
    restore_checkpoint,
    checkpoint_to_dict,
    checkpoint_from_dict,
)


def _make_engine(seed=42):
    engine = NeurochemicalEngine(dt=0.01, seed=seed)
    register_all_neurotransmitters(engine)
    return engine


class TestCheckpoint:
    def test_create_checkpoint(self):
        engine = _make_engine()
        for _ in range(5):
            engine.step()
        ckpt = create_checkpoint(engine, step_number=5)
        assert isinstance(ckpt, Checkpoint)
        assert ckpt.step_number == 5
        assert len(ckpt.neurotransmitter_states) == len(DEFAULT_NT_CONFIGS)

    def test_restore_checkpoint(self):
        engine = _make_engine()
        for _ in range(10):
            engine.step()

        ckpt = create_checkpoint(engine, step_number=10)

        # Mutate engine further
        for _ in range(10):
            engine.step()

        # Restore
        restore_checkpoint(engine, ckpt)
        assert engine.current_time == ckpt.current_time
        assert engine._step_number == 10

        for nt_name, state_dict in ckpt.neurotransmitter_states.items():
            state = engine.registry.get_neurotransmitter(nt_name)
            assert abs(state.C_tonic - state_dict["C_tonic"]) < 1e-12
            assert abs(state.C_phasic - state_dict["C_phasic"]) < 1e-12

    def test_checkpoint_round_trip_dict(self):
        engine = _make_engine()
        for _ in range(5):
            engine.step()
        ckpt = create_checkpoint(engine, step_number=5)
        data = checkpoint_to_dict(ckpt)
        ckpt2 = checkpoint_from_dict(data)

        assert ckpt2.step_number == ckpt.step_number
        assert ckpt2.current_time == ckpt.current_time
        assert set(ckpt2.neurotransmitter_states.keys()) == set(ckpt.neurotransmitter_states.keys())
        for nt_name in ckpt.neurotransmitter_states:
            for key in ("C_tonic", "C_phasic", "F"):
                assert ckpt2.neurotransmitter_states[nt_name][key] == ckpt.neurotransmitter_states[nt_name][key]

    def test_checkpoint_preserves_rng(self):
        engine = _make_engine(seed=42)
        register_all_neurotransmitters(engine)
        for _ in range(10):
            engine.step()

        ckpt = create_checkpoint(engine, step_number=10)

        # Run 5 more steps, record trajectory
        trajectory = []
        for _ in range(5):
            engine.step()
            da = engine.registry.get_neurotransmitter("DA")
            trajectory.append(da.C_tonic)

        # Restore checkpoint and replay
        restore_checkpoint(engine, ckpt)
        replay = []
        for _ in range(5):
            engine.step()
            da = engine.registry.get_neurotransmitter("DA")
            replay.append(da.C_tonic)

        assert trajectory == replay

    def test_checkpoint_preserves_time(self):
        engine = _make_engine()
        for _ in range(100):
            engine.step()
        ckpt = create_checkpoint(engine, step_number=100)
        engine.current_time = 0.0
        restore_checkpoint(engine, ckpt)
        assert abs(engine.current_time - 100 * 0.01) < 1e-10

    def test_checkpoint_with_receptors(self):
        engine = _make_engine()
        for _ in range(5):
            engine.step()
        ckpt = create_checkpoint(engine, step_number=5)

        total_receptors = sum(len(recs) for recs in NT_RECEPTOR_MAP.values())
        assert len(ckpt.receptor_states) == total_receptors

        for receptor_id, state_dict in ckpt.receptor_states.items():
            assert "rho" in state_dict
            assert "sigma" in state_dict
            assert "receptor_id" in state_dict
