"""Tests for deterministic seeding (Appendix N.5)."""

import numpy as np
import pytest

from zados.neurochem.optimization.seeding import (
    make_seed_sequence,
    derive_rng,
    create_rng_registry,
    save_rng_states,
    restore_rng_states,
    save_single_rng_state,
    restore_single_rng_state,
)
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.neurotransmitters.configs import register_all_neurotransmitters


class TestSeedSequence:
    def test_seed_sequence_creation(self):
        ss = make_seed_sequence(42)
        assert isinstance(ss, np.random.SeedSequence)
        assert ss.entropy == 42

    def test_derive_rng_different_streams(self):
        ss = make_seed_sequence(42)
        rng_a = derive_rng(ss, "dopamine_noise")
        rng_b = derive_rng(ss, "serotonin_noise")
        val_a = rng_a.normal(0, 1)
        val_b = rng_b.normal(0, 1)
        assert val_a != val_b

    def test_derive_rng_same_stream_same_seed(self):
        ss1 = make_seed_sequence(42)
        ss2 = make_seed_sequence(42)
        rng1 = derive_rng(ss1, "dopamine_noise", run_id=0)
        rng2 = derive_rng(ss2, "dopamine_noise", run_id=0)
        vals1 = [rng1.normal(0, 1) for _ in range(10)]
        vals2 = [rng2.normal(0, 1) for _ in range(10)]
        assert vals1 == vals2

    def test_derive_rng_different_run_ids(self):
        ss = make_seed_sequence(42)
        rng_r0 = derive_rng(ss, "DA", run_id=0)
        rng_r1 = derive_rng(ss, "DA", run_id=1)
        assert rng_r0.normal(0, 1) != rng_r1.normal(0, 1)

    def test_create_rng_registry(self):
        ss = make_seed_sequence(42)
        registry = create_rng_registry(ss, ["DA", "5HT", "NE"])
        assert set(registry.keys()) == {"DA", "5HT", "NE"}
        for rng in registry.values():
            assert isinstance(rng, np.random.Generator)

    def test_save_restore_rng_states(self):
        ss = make_seed_sequence(42)
        registry = create_rng_registry(ss, ["DA", "5HT"])

        # Draw some values to advance state
        registry["DA"].normal(0, 1)
        registry["5HT"].normal(0, 1)

        # Save state
        saved = save_rng_states(registry)

        # Draw more values
        expected_da = registry["DA"].normal(0, 1)
        expected_5ht = registry["5HT"].normal(0, 1)

        # Restore and verify same values
        restore_rng_states(registry, saved)
        assert registry["DA"].normal(0, 1) == expected_da
        assert registry["5HT"].normal(0, 1) == expected_5ht


class TestSingleRNGStateOps:
    def test_save_restore_single(self):
        rng = np.random.default_rng(123)
        rng.normal(0, 1)  # advance
        state = save_single_rng_state(rng)
        expected = rng.normal(0, 1)
        restore_single_rng_state(rng, state)
        assert rng.normal(0, 1) == expected


class TestEngineRNG:
    def test_engine_rng_reproducibility(self):
        """Same seed → identical step() output."""
        engine1 = NeurochemicalEngine(dt=0.01, seed=42)
        engine2 = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine1)
        register_all_neurotransmitters(engine2)

        for _ in range(10):
            engine1.step()
            engine2.step()

        for nt_name in engine1.registry.neurotransmitter_names():
            s1 = engine1.registry.get_neurotransmitter(nt_name)
            s2 = engine2.registry.get_neurotransmitter(nt_name)
            assert s1.C_tonic == s2.C_tonic, f"{nt_name} C_tonic mismatch"
            assert s1.C_phasic == s2.C_phasic, f"{nt_name} C_phasic mismatch"

    def test_engine_rng_different_seeds(self):
        """Different seeds → different output."""
        engine1 = NeurochemicalEngine(dt=0.01, seed=42)
        engine2 = NeurochemicalEngine(dt=0.01, seed=99)
        register_all_neurotransmitters(engine1)
        register_all_neurotransmitters(engine2)

        for _ in range(10):
            engine1.step()
            engine2.step()

        # At least one NT should differ
        any_different = False
        for nt_name in engine1.registry.neurotransmitter_names():
            s1 = engine1.registry.get_neurotransmitter(nt_name)
            s2 = engine2.registry.get_neurotransmitter(nt_name)
            if s1.C_tonic != s2.C_tonic:
                any_different = True
                break
        assert any_different

    def test_engine_has_numpy_rng(self):
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        assert isinstance(engine.rng, np.random.Generator)

    def test_engine_step_number_increments(self):
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine)
        assert engine._step_number == 0
        engine.step()
        assert engine._step_number == 1
        engine.step()
        assert engine._step_number == 2
