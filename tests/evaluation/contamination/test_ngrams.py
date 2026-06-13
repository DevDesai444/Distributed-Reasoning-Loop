"""Unit tests for ngram extraction."""

from __future__ import annotations

import pytest

from src.evaluation.contamination.ngrams import ngram_set, ngrams, normalize


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize("  Hello   World\n") == "hello world"


def test_normalize_strips_punctuation_but_keeps_digits():
    assert normalize("Natalia sold $48, in April!") == "natalia sold 48 in april"


def test_normalize_empty_string():
    assert normalize("") == ""
    assert normalize(None) == ""  # type: ignore[arg-type]


def test_ngrams_basic():
    grams = list(ngrams("the quick brown fox jumps", n=3))
    assert grams == [
        ("the", "quick", "brown"),
        ("quick", "brown", "fox"),
        ("brown", "fox", "jumps"),
    ]


def test_ngrams_shorter_than_n_yields_nothing():
    assert list(ngrams("short text", n=5)) == []


def test_ngrams_exactly_n_tokens_yields_one():
    grams = list(ngrams("a b c d", n=4))
    assert grams == [("a", "b", "c", "d")]


def test_ngrams_empty_text_yields_nothing():
    assert list(ngrams("", n=3)) == []
    assert list(ngrams("   ", n=3)) == []


def test_ngrams_rejects_zero_n():
    with pytest.raises(ValueError):
        list(ngrams("hello world there", n=0))


def test_ngram_set_unions_across_documents():
    s = ngram_set(["a b c", "a b c d"], n=2)
    assert ("a", "b") in s
    assert ("b", "c") in s
    assert ("c", "d") in s
    assert len(s) == 3


def test_normalize_preserves_digits_for_math_problems():
    # Critical for GSM8K-style problems where numbers carry identity.
    assert "48" in normalize("She sold 48 clips.")
    # Comma turns into a word separator. That's fine for n-gram matching: both
    # "1,234" and "1 234" normalise to the same token sequence.
    assert normalize("answer is 1,234") == "answer is 1 234"
