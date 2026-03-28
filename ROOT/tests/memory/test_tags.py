"""Tests for LTMM tag taxonomy."""
import pytest
from zados.memory.long_term.tags import (
    ALL_SYSTEM_TAGS, IDENTITY_TAGS, COGNITIVE_TAGS, PIPELINE_TAGS,
    DOMAIN_TAGS, TAG_PREFIXES, validate_tags,
)


class TestTagSets:
    def test_identity_tags_have_prefix(self):
        for tag in IDENTITY_TAGS:
            assert tag.startswith("identity:")

    def test_cognitive_tags_have_prefix(self):
        for tag in COGNITIVE_TAGS:
            assert tag.startswith("cognitive:")

    def test_pipeline_tags_have_prefix(self):
        for tag in PIPELINE_TAGS:
            assert tag.startswith("pipeline:")

    def test_domain_tags_have_prefix(self):
        for tag in DOMAIN_TAGS:
            assert tag.startswith("domain:")

    def test_all_system_tags_is_union(self):
        assert ALL_SYSTEM_TAGS == IDENTITY_TAGS | COGNITIVE_TAGS | PIPELINE_TAGS | DOMAIN_TAGS

    def test_no_duplicates_across_sets(self):
        total = len(IDENTITY_TAGS) + len(COGNITIVE_TAGS) + len(PIPELINE_TAGS) + len(DOMAIN_TAGS)
        assert total == len(ALL_SYSTEM_TAGS)


class TestValidateTags:
    def test_valid_system_tags(self):
        assert validate_tags(["identity:core", "pipeline:m1"]) == []

    def test_custom_tags_always_valid(self):
        assert validate_tags(["my_custom_tag", "another:custom"]) == []

    def test_invalid_system_tag(self):
        result = validate_tags(["identity:nonexistent"])
        assert "identity:nonexistent" in result

    def test_mixed_valid_and_invalid(self):
        result = validate_tags(["identity:core", "cognitive:invalid_one", "custom_ok"])
        assert len(result) == 1
        assert result[0] == "cognitive:invalid_one"

    def test_empty_list(self):
        assert validate_tags([]) == []
