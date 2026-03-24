"""Tests for shared TF-IDF search utilities."""
import pytest
from zados.memory.long_term.search_utils import tokenize, term_freq, cosine


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_empty(self):
        assert tokenize("") == []

    def test_strips_punctuation(self):
        tokens = tokenize("foo-bar, baz! qux.")
        assert "foo" in tokens
        assert "bar" in tokens

    def test_preserves_numbers(self):
        assert tokenize("v2 beta3") == ["v2", "beta3"]


class TestTermFreq:
    def test_uniform(self):
        tf = term_freq(["a", "b", "c"])
        assert abs(tf["a"] - 1 / 3) < 1e-9

    def test_repeated(self):
        tf = term_freq(["a", "a", "b"])
        assert abs(tf["a"] - 2 / 3) < 1e-9

    def test_empty(self):
        tf = term_freq([])
        assert tf == {}


class TestCosine:
    def test_identical(self):
        v = {"a": 1.0, "b": 2.0}
        assert abs(cosine(v, v) - 1.0) < 1e-9

    def test_disjoint(self):
        assert cosine({"a": 1.0}, {"b": 1.0}) == 0.0

    def test_empty(self):
        assert cosine({}, {"a": 1.0}) == 0.0

    def test_partial_overlap(self):
        a = {"x": 1.0, "y": 1.0}
        b = {"y": 1.0, "z": 1.0}
        sim = cosine(a, b)
        assert 0.0 < sim < 1.0
