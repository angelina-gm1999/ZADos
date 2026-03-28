"""
Tests for E31/E32 wiring into constants.py and engine_toolkit.py.
=================================================================
Verifies that the reflective learning and identity engines are
correctly registered in ENGINE_IDS, ENGINE_CLUSTER_MAP, BASE_TIERS,
and removed from PHANTOM_ENGINES.
"""
from __future__ import annotations

import pytest

from zados.cognitive_engines.constants import ENGINE_IDS, ENGINE_CLUSTER_MAP
from zados.core.processes.engine_toolkit import (
    ALL_ENGINE_IDS,
    BASE_TIERS,
    BUDGET_CAPS,
    PHANTOM_ENGINES,
    PHANTOM_ENGINE_IDS,
    EngineToolkit,
)
from zados.core.types import EngineTier, SubjectCategory


# =====================================================================
# Constants registration
# =====================================================================

class TestConstantsRegistration:
    def test_e31_in_engine_ids(self):
        assert 31 in ENGINE_IDS
        assert ENGINE_IDS[31] == "reflective_learning_engine"

    def test_e32_in_engine_ids(self):
        assert 32 in ENGINE_IDS
        assert ENGINE_IDS[32] == "reflective_identity_engine"

    def test_e31_cluster_metacognition(self):
        assert ENGINE_CLUSTER_MAP[31] == "metacognition"

    def test_e32_cluster_metacognition(self):
        assert ENGINE_CLUSTER_MAP[32] == "metacognition"

    def test_engine_ids_consistency(self):
        """All IDs in ENGINE_IDS should also be in ENGINE_CLUSTER_MAP."""
        for eid in ENGINE_IDS:
            assert eid in ENGINE_CLUSTER_MAP, f"E{eid} missing from cluster map"


# =====================================================================
# Phantom engine removal
# =====================================================================

class TestPhantomEngineRemoval:
    def test_e31_not_phantom(self):
        assert "reflective_learning_engine" not in PHANTOM_ENGINES

    def test_e32_not_phantom(self):
        assert "reflective_identity_engine" not in PHANTOM_ENGINES

    def test_e31_not_in_phantom_ids(self):
        assert 31 not in PHANTOM_ENGINE_IDS

    def test_e32_not_in_phantom_ids(self):
        assert 32 not in PHANTOM_ENGINE_IDS

    def test_emotional_saturation_still_phantom(self):
        assert "emotional_saturation_engine" in PHANTOM_ENGINES

    def test_e33_still_in_phantom_ids(self):
        assert 33 in PHANTOM_ENGINE_IDS


# =====================================================================
# ALL_ENGINE_IDS
# =====================================================================

class TestAllEngineIds:
    def test_includes_e31(self):
        assert 31 in ALL_ENGINE_IDS

    def test_includes_e32(self):
        assert 32 in ALL_ENGINE_IDS

    def test_includes_phantoms(self):
        for eid in PHANTOM_ENGINE_IDS:
            assert eid in ALL_ENGINE_IDS


# =====================================================================
# Reflective mode in BASE_TIERS
# =====================================================================

class TestReflectiveBaseTiers:
    def test_reflective_mode_exists(self):
        assert "reflective" in BASE_TIERS

    def test_e31_t1_in_reflective(self):
        tiers = BASE_TIERS["reflective"]
        assert tiers["reflective_learning_engine"] == EngineTier.T1

    def test_e32_t1_in_reflective(self):
        tiers = BASE_TIERS["reflective"]
        assert tiers["reflective_identity_engine"] == EngineTier.T1

    def test_detection_engines_t3_or_lower(self):
        """Detection engines should be T3 standby in reflective mode."""
        tiers = BASE_TIERS["reflective"]
        detection_engines = [
            "contradiction_detection_engine",
            "paradox_detection_engine",
            "fallacy_detection_engine",
            "bias_detection_engine",
        ]
        for eng in detection_engines:
            assert tiers[eng].value >= EngineTier.T3.value, (
                f"{eng} should be T3 or T4 in reflective mode"
            )

    def test_no_user_engines_off(self):
        """Engines requiring user presence should be T4 off."""
        tiers = BASE_TIERS["reflective"]
        assert tiers["intention_map_engine"] == EngineTier.T4
        assert tiers["emotional_detection_engine"] == EngineTier.T4

    def test_learning_engines_t1(self):
        """Learning engines should be T1 for E31 feed."""
        tiers = BASE_TIERS["reflective"]
        learning_engines = [
            "pattern_identification_engine",
            "pattern_comparison_engine",
            "contextual_learning_engine",
            "recursive_learning_engine",
            "reward_based_learning_engine",
        ]
        for eng in learning_engines:
            assert tiers[eng] == EngineTier.T1, (
                f"{eng} should be T1 in reflective mode"
            )

    def test_homeostatic_t1(self):
        tiers = BASE_TIERS["reflective"]
        assert tiers["neurochemical_homeostatic_engine"] == EngineTier.T1


# =====================================================================
# Budget caps
# =====================================================================

class TestBudgetCaps:
    def test_reflective_budget_exists(self):
        assert "reflective" in BUDGET_CAPS

    def test_reflective_budget_value(self):
        assert BUDGET_CAPS["reflective"] == 12

    def test_reflective_budget_enforced(self):
        tk = EngineToolkit()
        tiers = tk.resolve("reflective", SubjectCategory.MIXED)
        active_count = sum(
            1 for t in tiers.values()
            if t in (EngineTier.T1, EngineTier.T2)
        )
        assert active_count <= BUDGET_CAPS["reflective"]


# =====================================================================
# EngineToolkit.resolve — reflective mode
# =====================================================================

class TestEngineToolkitResolve:
    def test_resolve_reflective(self):
        tk = EngineToolkit()
        tiers = tk.resolve("reflective")
        assert isinstance(tiers, dict)
        assert "reflective_learning_engine" in tiers
        assert "reflective_identity_engine" in tiers

    def test_e31_not_phantom_in_resolve(self):
        """E31 should NOT be forced to T4 by phantom logic."""
        tk = EngineToolkit()
        tiers = tk.resolve("reflective")
        assert tiers["reflective_learning_engine"] == EngineTier.T1

    def test_e32_not_phantom_in_resolve(self):
        tk = EngineToolkit()
        tiers = tk.resolve("reflective")
        assert tiers["reflective_identity_engine"] == EngineTier.T1

    def test_other_modes_have_e31_e32(self):
        """E31/E32 should appear in other modes too (default T3)."""
        tk = EngineToolkit()
        for mode in ("regular", "M1", "M2", "M3", "M4", "M5"):
            tiers = tk.resolve(mode)
            assert "reflective_learning_engine" in tiers, (
                f"E31 missing from {mode} tiers"
            )
            assert "reflective_identity_engine" in tiers, (
                f"E32 missing from {mode} tiers"
            )

    def test_e31_e32_default_tier_in_regular(self):
        """In regular mode, E31/E32 should default to T3 (not phantom T4)."""
        tk = EngineToolkit()
        tiers = tk.resolve("regular")
        # Since they're no longer phantom, they should get the default T3
        assert tiers["reflective_learning_engine"] == EngineTier.T3
        assert tiers["reflective_identity_engine"] == EngineTier.T3

    def test_tiers_to_weights(self):
        tk = EngineToolkit()
        tiers = tk.resolve("reflective")
        weights = tk.tiers_to_weights(tiers)
        assert weights["reflective_learning_engine"] == 1.0  # T1 → 1.0
        assert weights["reflective_identity_engine"] == 1.0  # T1 → 1.0

    def test_tiers_to_weights_by_id(self):
        tk = EngineToolkit()
        tiers = tk.resolve("reflective")
        weights = tk.tiers_to_weights_by_id(tiers)
        assert weights["31"] == 1.0
        assert weights["32"] == 1.0
