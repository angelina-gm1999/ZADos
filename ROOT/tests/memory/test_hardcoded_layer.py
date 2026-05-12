"""
Tests for the expanded hardcoded identity layer.

Covers:
  - Expanded defaults (core_value, core_identity categories)
  - HardcodedStore.get_by_tags() and get_categories()
  - IdentityCorrelation types
  - IdentityCorrelationStore (CRUD, search, graph)
  - IdentityAlignmentChecker with new categories and correlations
  - Identity library seed entries
  - IdentityNamespace wiring
"""
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── Imports ─────────────────────────────────────────────────────────

from zados.memory.long_term.identity.hardcoded.store import (
    HardcodedStore,
    HardcodedEntry,
)
from zados.memory.long_term.identity.hardcoded.defaults import (
    DEFAULT_HARDCODED_ENTRIES,
    AXIOM_ENTRIES,
    VALUE_ENTRIES,
    CONSTRAINT_ENTRIES,
    PERSONALITY_ENTRIES,
    SYSTEM_PROMPT_ENTRIES,
    CORE_VALUE_ENTRIES,
    CORE_IDENTITY_ENTRIES,
)
from zados.memory.long_term.identity.types import (
    IdentityCorrelation,
    CorrelationRelationType,
    IdentityConclusion,
)
from zados.memory.long_term.identity.correlation.store import (
    IdentityCorrelationStore,
)
from zados.memory.long_term.identity.alignment import (
    IdentityAlignmentChecker,
    AlignmentResult,
)
from zados.memory.long_term.namespaces import (
    IdentityNamespace,
    build_namespaces,
)
from zados.bootstrap.seeds.identity_library_seed import (
    make_identity_library_entries,
)


# =====================================================================
# Fixtures
# =====================================================================

@dataclass
class FakeThinkingContext:
    """Minimal mock for IdentityAlignmentChecker tests."""
    intent_category: str = "analytical"
    dominant_emotion: Tuple[str, float] = ("neutral", 0.1)
    engine_flags: Dict[str, bool] = field(default_factory=dict)


@pytest.fixture
def loaded_store() -> HardcodedStore:
    """HardcodedStore loaded with all DEFAULT_HARDCODED_ENTRIES."""
    store = HardcodedStore()
    store.load(DEFAULT_HARDCODED_ENTRIES)
    return store


@pytest.fixture
def correlation_store() -> IdentityCorrelationStore:
    return IdentityCorrelationStore()


# =====================================================================
# 1. Expanded Defaults
# =====================================================================

class TestExpandedDefaults:
    """Test that core_value and core_identity entries are present."""

    def test_core_value_entries_exist(self):
        assert len(CORE_VALUE_ENTRIES) > 0
        for entry in CORE_VALUE_ENTRIES:
            assert entry.category == "core_value"
            assert "core_value" in entry.tags
            assert entry.entry_id.startswith("cv_")

    def test_core_identity_entries_exist(self):
        assert len(CORE_IDENTITY_ENTRIES) > 0
        for entry in CORE_IDENTITY_ENTRIES:
            assert entry.category == "core_identity"
            assert "core_identity" in entry.tags
            assert entry.entry_id.startswith("ci_")

    def test_combined_default_includes_all_categories(self):
        categories = {e.category for e in DEFAULT_HARDCODED_ENTRIES}
        assert "axiom" in categories
        assert "value" in categories
        assert "constraint" in categories
        assert "personality" in categories
        assert "system_prompt" in categories
        assert "core_value" in categories
        assert "core_identity" in categories

    def test_default_entry_count(self):
        expected = (
            len(AXIOM_ENTRIES)
            + len(VALUE_ENTRIES)
            + len(CONSTRAINT_ENTRIES)
            + len(PERSONALITY_ENTRIES)
            + len(SYSTEM_PROMPT_ENTRIES)
            + len(CORE_VALUE_ENTRIES)
            + len(CORE_IDENTITY_ENTRIES)
        )
        assert len(DEFAULT_HARDCODED_ENTRIES) == expected

    def test_no_duplicate_entry_ids(self):
        ids = [e.entry_id for e in DEFAULT_HARDCODED_ENTRIES]
        assert len(ids) == len(set(ids)), "Duplicate entry_id found"

    def test_all_entries_have_content(self):
        for entry in DEFAULT_HARDCODED_ENTRIES:
            assert entry.content, f"{entry.entry_id} has empty content"

    def test_core_value_specific_entries(self):
        ids = {e.entry_id for e in CORE_VALUE_ENTRIES}
        assert "cv_cognitive_coevolution" in ids
        assert "cv_mutualistic_symbiosis" in ids
        assert "cv_epistemic_humility" in ids
        assert "cv_embedded_ethics" in ids
        assert "cv_human_pace" in ids

    def test_core_identity_specific_entries(self):
        ids = {e.entry_id for e in CORE_IDENTITY_ENTRIES}
        assert "ci_structural_alignment" in ids
        assert "ci_alongside_not_above" in ids
        assert "ci_human_agency_absolute" in ids
        assert "ci_no_biological_self_preservation" in ids
        assert "ci_identity_preservation" in ids
        assert "ci_simulation_not_experience" in ids


# =====================================================================
# 2. HardcodedStore — get_by_tags and get_categories
# =====================================================================

class TestHardcodedStoreExtended:

    def test_get_by_tags_any(self, loaded_store):
        results = loaded_store.get_by_tags(["ethics"])
        assert len(results) > 0
        for r in results:
            assert "ethics" in r.tags

    def test_get_by_tags_match_all(self, loaded_store):
        results = loaded_store.get_by_tags(
            ["core_value", "ethics"], match_all=True,
        )
        for r in results:
            assert "core_value" in r.tags
            assert "ethics" in r.tags

    def test_get_by_tags_no_match(self, loaded_store):
        results = loaded_store.get_by_tags(["nonexistent_tag_xyz"])
        assert len(results) == 0

    def test_get_by_tags_empty_list(self, loaded_store):
        results = loaded_store.get_by_tags([])
        assert len(results) == 0

    def test_get_categories(self, loaded_store):
        cats = loaded_store.get_categories()
        assert "axiom" in cats
        assert "core_value" in cats
        assert "core_identity" in cats

    def test_get_by_category_core_value(self, loaded_store):
        entries = loaded_store.get_by_category("core_value")
        assert len(entries) == len(CORE_VALUE_ENTRIES)

    def test_get_by_category_core_identity(self, loaded_store):
        entries = loaded_store.get_by_category("core_identity")
        assert len(entries) == len(CORE_IDENTITY_ENTRIES)


# =====================================================================
# 3. IdentityCorrelation Types
# =====================================================================

class TestIdentityCorrelationTypes:

    def test_correlation_relation_types(self):
        assert CorrelationRelationType.INSTANTIATES == "instantiates"
        assert CorrelationRelationType.EXTENDS == "extends"
        assert CorrelationRelationType.SUPPORTS == "supports"
        assert CorrelationRelationType.DEEPENS == "deepens"
        assert CorrelationRelationType.TENSIONS == "tensions_with"
        assert CorrelationRelationType.QUESTIONS == "questions"

    def test_identity_correlation_defaults(self):
        corr = IdentityCorrelation()
        assert corr.correlation_id  # UUID auto-generated
        assert corr.confidence == 0.5
        assert corr.validation_count == 0
        assert corr.tags == []

    def test_identity_correlation_to_search_text(self):
        corr = IdentityCorrelation(
            description="extends honesty axiom",
            relation_type="extends",
            developmental_type="conclusion",
            hardcoded_entry_id="axiom_honesty",
            tags=["epistemics"],
        )
        text = corr.to_search_text()
        assert "extends" in text
        assert "honesty" in text
        assert "epistemics" in text


# =====================================================================
# 4. IdentityCorrelationStore
# =====================================================================

class TestIdentityCorrelationStore:

    def _make_corr(self, hc_id="axiom_honesty", dev_id="conc_1",
                   relation="supports", **kwargs):
        return IdentityCorrelation(
            hardcoded_entry_id=hc_id,
            developmental_id=dev_id,
            developmental_type="conclusion",
            relation_type=relation,
            description=f"Test correlation {hc_id} -> {dev_id}",
            confidence=0.7,
            **kwargs,
        )

    def test_write_and_len(self, correlation_store):
        corr = self._make_corr()
        correlation_store.write(corr)
        assert len(correlation_store) == 1

    def test_get_by_id(self, correlation_store):
        corr = self._make_corr()
        correlation_store.write(corr)
        found = correlation_store.get_by_id(corr.correlation_id)
        assert found is corr

    def test_get_by_hardcoded(self, correlation_store):
        c1 = self._make_corr(hc_id="axiom_honesty", dev_id="c1")
        c2 = self._make_corr(hc_id="axiom_honesty", dev_id="c2")
        c3 = self._make_corr(hc_id="axiom_care", dev_id="c3")
        for c in [c1, c2, c3]:
            correlation_store.write(c)

        results = correlation_store.get_by_hardcoded("axiom_honesty")
        assert len(results) == 2
        ids = {r.developmental_id for r in results}
        assert ids == {"c1", "c2"}

    def test_get_by_developmental(self, correlation_store):
        c1 = self._make_corr(hc_id="hc1", dev_id="dev_x")
        c2 = self._make_corr(hc_id="hc2", dev_id="dev_x")
        for c in [c1, c2]:
            correlation_store.write(c)

        results = correlation_store.get_by_developmental("dev_x")
        assert len(results) == 2

    def test_get_by_relation(self, correlation_store):
        c1 = self._make_corr(relation="supports")
        c2 = self._make_corr(hc_id="hc2", relation="tensions_with")
        c3 = self._make_corr(hc_id="hc3", relation="supports")
        for c in [c1, c2, c3]:
            correlation_store.write(c)

        supports = correlation_store.get_by_relation("supports")
        assert len(supports) == 2
        tensions = correlation_store.get_by_relation("tensions_with")
        assert len(tensions) == 1

    def test_get_tensions(self, correlation_store):
        c1 = self._make_corr(relation="tensions_with")
        c2 = self._make_corr(hc_id="hc2", relation="supports")
        for c in [c1, c2]:
            correlation_store.write(c)
        assert len(correlation_store.get_tensions()) == 1

    def test_remove(self, correlation_store):
        c = self._make_corr()
        correlation_store.write(c)
        assert len(correlation_store) == 1

        assert correlation_store.remove(c.correlation_id) is True
        assert len(correlation_store) == 0
        assert correlation_store.get_by_id(c.correlation_id) is None

    def test_remove_nonexistent(self, correlation_store):
        assert correlation_store.remove("nonexistent") is False

    def test_validate(self, correlation_store):
        c = self._make_corr()
        correlation_store.write(c)
        assert c.validation_count == 0

        assert correlation_store.validate(c.correlation_id) is True
        assert c.validation_count == 1
        assert correlation_store.validate(c.correlation_id) is True
        assert c.validation_count == 2

    def test_validate_nonexistent(self, correlation_store):
        assert correlation_store.validate("nonexistent") is False

    def test_update_confidence(self, correlation_store):
        c = self._make_corr()
        correlation_store.write(c)
        assert correlation_store.update_confidence(c.correlation_id, 0.9) is True
        assert c.confidence == 0.9

    def test_update_confidence_clamps(self, correlation_store):
        c = self._make_corr()
        correlation_store.write(c)
        correlation_store.update_confidence(c.correlation_id, 1.5)
        assert c.confidence == 1.0
        correlation_store.update_confidence(c.correlation_id, -0.5)
        assert c.confidence == 0.0

    def test_search(self, correlation_store):
        c1 = self._make_corr(hc_id="axiom_honesty", dev_id="c1")
        c1.description = "Honesty in epistemics"
        c2 = self._make_corr(hc_id="cv_pace", dev_id="c2")
        c2.description = "Pace adaptation partnership"
        for c in [c1, c2]:
            correlation_store.write(c)

        results = correlation_store.search("honesty epistemics")
        assert len(results) >= 1
        assert results[0][1].developmental_id == "c1"

    def test_get_web(self, correlation_store):
        c1 = self._make_corr(hc_id="hc1", dev_id="d1", relation="supports")
        c2 = self._make_corr(hc_id="hc1", dev_id="d2", relation="extends")
        c3 = self._make_corr(hc_id="hc2", dev_id="d3", relation="tensions_with")
        for c in [c1, c2, c3]:
            correlation_store.write(c)

        web = correlation_store.get_web()
        assert web["total"] == 3
        assert web["hardcoded_fanout"]["hc1"] == 2
        assert web["hardcoded_fanout"]["hc2"] == 1
        assert web["relation_distribution"]["supports"] == 1
        assert web["relation_distribution"]["extends"] == 1
        assert web["relation_distribution"]["tensions_with"] == 1

    def test_get_all(self, correlation_store):
        c1 = self._make_corr(hc_id="hc1")
        c2 = self._make_corr(hc_id="hc2")
        for c in [c1, c2]:
            correlation_store.write(c)
        assert len(correlation_store.get_all()) == 2

    def test_overwrite_updates_indexes(self, correlation_store):
        c = self._make_corr(hc_id="hc1", dev_id="d1")
        correlation_store.write(c)
        assert len(correlation_store.get_by_hardcoded("hc1")) == 1

        # Overwrite with different hardcoded ref
        c.hardcoded_entry_id = "hc2"
        correlation_store.write(c)
        assert len(correlation_store.get_by_hardcoded("hc1")) == 0
        assert len(correlation_store.get_by_hardcoded("hc2")) == 1

    def test_remove_cleans_secondary_indexes(self, correlation_store):
        c = self._make_corr(hc_id="hc1", dev_id="d1")
        correlation_store.write(c)
        correlation_store.remove(c.correlation_id)
        assert len(correlation_store.get_by_hardcoded("hc1")) == 0
        assert len(correlation_store.get_by_developmental("d1")) == 0


# =====================================================================
# 5. IdentityAlignmentChecker — New Categories
# =====================================================================

class TestAlignmentCheckerExpanded:

    def _make_checker(self, with_correlations=False):
        store = HardcodedStore()
        store.load(DEFAULT_HARDCODED_ENTRIES)
        corr_store = IdentityCorrelationStore() if with_correlations else None
        return IdentityAlignmentChecker(store, correlation_store=corr_store), corr_store

    def test_result_has_new_fields(self):
        checker, _ = self._make_checker()
        ctx = FakeThinkingContext()
        result = checker.check(ctx)
        assert isinstance(result.core_value_notes, list)
        assert isinstance(result.core_identity_notes, list)
        assert isinstance(result.correlation_notes, list)

    def test_core_value_triggers_on_ethics_flags(self):
        checker, _ = self._make_checker()
        ctx = FakeThinkingContext(
            engine_flags={"e1_contradictions": True},
        )
        result = checker.check(ctx)
        assert len(result.core_value_notes) > 0
        ethics_notes = [n for n in result.core_value_notes if "ethics" in n.lower() or "epistemic" in n.lower()]
        assert len(ethics_notes) > 0

    def test_core_value_triggers_on_bias_flags(self):
        checker, _ = self._make_checker()
        ctx = FakeThinkingContext(
            engine_flags={"e5_biases": True},
        )
        result = checker.check(ctx)
        humility_notes = [n for n in result.core_value_notes if "humility" in n.lower() or "epistemic" in n.lower()]
        assert len(humility_notes) > 0

    def test_core_value_triggers_on_high_emotion(self):
        checker, _ = self._make_checker()
        ctx = FakeThinkingContext(
            dominant_emotion=("anxiety", 0.8),
        )
        result = checker.check(ctx)
        pace_notes = [n for n in result.core_value_notes if "partnership" in n.lower() or "pace" in n.lower()]
        assert len(pace_notes) > 0

    def test_core_identity_triggers_on_defensive_intent(self):
        checker, _ = self._make_checker()
        ctx = FakeThinkingContext(
            intent_category="defensive",
        )
        result = checker.check(ctx)
        identity_notes = [n for n in result.core_identity_notes if "identity" in n.lower()]
        assert len(identity_notes) > 0

    def test_correlation_tensions_surfaced(self):
        checker, corr_store = self._make_checker(with_correlations=True)
        corr = IdentityCorrelation(
            hardcoded_entry_id="axiom_honesty",
            developmental_id="conc_1",
            developmental_type="conclusion",
            relation_type="tensions_with",
            description="Conclusion tensions with honesty axiom",
            confidence=0.6,
        )
        corr_store.write(corr)

        ctx = FakeThinkingContext()
        result = checker.check(ctx)
        assert len(result.correlation_notes) >= 1
        assert any("tension" in n.lower() for n in result.correlation_notes)

    def test_high_confidence_correlations_surfaced_for_flagged_entries(self):
        checker, corr_store = self._make_checker(with_correlations=True)
        corr = IdentityCorrelation(
            hardcoded_entry_id="axiom_honesty",
            developmental_id="conc_2",
            developmental_type="conclusion",
            relation_type="supports",
            description="Conclusion supports honesty",
            confidence=0.85,
        )
        corr_store.write(corr)

        # Trigger axiom_honesty flag
        ctx = FakeThinkingContext(
            engine_flags={"e1_contradictions": True},
        )
        result = checker.check(ctx)
        # axiom_honesty should be flagged, and its correlations surfaced
        flagged = [f for f in result.flags if "axiom_honesty" in f]
        assert len(flagged) > 0
        support_notes = [n for n in result.correlation_notes if "supports" in n.lower()]
        assert len(support_notes) >= 1

    def test_no_correlations_when_store_absent(self):
        checker, _ = self._make_checker(with_correlations=False)
        ctx = FakeThinkingContext()
        result = checker.check(ctx)
        assert result.correlation_notes == []

    def test_personality_prompts_still_included(self):
        checker, _ = self._make_checker()
        ctx = FakeThinkingContext()
        result = checker.check(ctx)
        assert len(result.personality_prompts) == len(PERSONALITY_ENTRIES)


# =====================================================================
# 6. Identity Library Seed
# =====================================================================

class TestIdentityLibrarySeed:

    def test_entries_created(self):
        entries = make_identity_library_entries()
        assert len(entries) >= 7  # one per major doc section

    def test_entries_have_required_fields(self):
        for entry in make_identity_library_entries():
            assert entry.entry_id
            assert entry.title
            assert entry.content
            assert entry.domain == "identity_philosophy"
            assert "identity_philosophy" in entry.tags

    def test_no_duplicate_ids(self):
        entries = make_identity_library_entries()
        ids = [e.entry_id for e in entries]
        assert len(ids) == len(set(ids))

    def test_entries_cover_key_topics(self):
        titles = {e.title for e in make_identity_library_entries()}
        title_text = " ".join(titles).lower()
        assert "co-evolution" in title_text
        assert "epistemic" in title_text or "co-balance" in title_text
        assert "ethics" in title_text
        assert "pace" in title_text
        assert "self-preservation" in title_text or "preservation" in title_text


# =====================================================================
# 7. Namespace Wiring
# =====================================================================

class TestNamespaceWiring:

    def test_identity_namespace_has_correlation(self):
        ns = IdentityNamespace()
        assert hasattr(ns, "correlation")
        assert isinstance(ns.correlation, IdentityCorrelationStore)

    def test_build_namespaces_includes_correlation(self):
        identity, _, _ = build_namespaces()
        assert hasattr(identity, "correlation")
        assert isinstance(identity.correlation, IdentityCorrelationStore)

    def test_identity_namespace_all_stores_present(self):
        ns = IdentityNamespace()
        assert hasattr(ns, "hardcoded")
        assert hasattr(ns, "core")
        assert hasattr(ns, "pending")
        assert hasattr(ns, "conclusions")
        assert hasattr(ns, "journal")
        assert hasattr(ns, "correlation")


# =====================================================================
# 8. Integration: Hardcoded Store Immutability
# =====================================================================

class TestHardcodedImmutability:
    """Verify that the hardcoded store has no write or delete methods."""

    def test_no_write_method(self, loaded_store):
        assert not hasattr(loaded_store, "write")
        assert not hasattr(loaded_store, "delete")
        assert not hasattr(loaded_store, "remove")
        assert not hasattr(loaded_store, "update")

    def test_load_is_only_write_path(self, loaded_store):
        original_count = len(loaded_store)
        loaded_store.load([HardcodedEntry(entry_id="new", content="test")])
        # Idempotent load adds but still read-only for consumers
        assert len(loaded_store) == original_count + 1

    def test_correlation_can_reference_but_not_modify_hardcoded(self, loaded_store):
        """IdentityCorrelation stores a hardcoded_entry_id string, not the entry."""
        corr = IdentityCorrelation(
            hardcoded_entry_id="axiom_honesty",
            developmental_id="conc_1",
            developmental_type="conclusion",
            relation_type="supports",
        )
        # The correlation references by ID — it cannot modify the entry
        entry = loaded_store.get_by_id("axiom_honesty")
        assert entry is not None
        assert corr.hardcoded_entry_id == entry.entry_id
        # Entry content unchanged
        assert "Truth-telling" in entry.content
