"""
Token n-gram extraction for contamination analysis.

Why 13-grams by default: long enough that incidental overlap between unrelated
text is vanishingly rare, short enough that fragmented quotations still trip.
This is the conventional unit in large-corpus contamination audits across the
literature. Punctuation is stripped but digits are kept because in math
benchmarks the specific numbers in a problem are most of its identity.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, Tuple

# Keep digits, letters, whitespace. Drop everything else so that "$48," and "48"
# normalise to the same token. We intentionally do not lowercase digits (n/a)
# but do lowercase letters so "Natalia" and "natalia" match.
_NON_TOKEN = re.compile(r"[^0-9a-zA-Z\s]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip non-alphanumeric punctuation, collapse whitespace.

    We keep digits because numeric content carries problem identity in math
    benchmarks. We do not stem or strip stopwords — 13-gram windows already
    swamp the signal from small function-word substitutions.
    """
    if not text:
        return ""
    lowered = text.lower()
    stripped = _NON_TOKEN.sub(" ", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()


def ngrams(text: str, n: int = 13) -> Iterator[Tuple[str, ...]]:
    """Yield every contiguous n-gram of whitespace tokens from ``text``.

    Yields nothing for empty input or input with fewer than ``n`` tokens.
    Generator-based so large corpora stream through without materialising
    a full token list per document twice.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    normalized = normalize(text)
    if not normalized:
        return
    tokens = normalized.split(" ")
    if len(tokens) < n:
        return
    # Pre-cache tuple slicing in a tight loop. tuple() is the cheapest hashable
    # window we can build.
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def ngram_set(texts: Iterable[str], n: int = 13) -> set[Tuple[str, ...]]:
    """Collect the union of n-grams across an iterable of documents.

    Use this when you want the haystack as a single membership-test set. For
    very large corpora the set can get large (millions of tuples) but each
    tuple is short strings, so memory pressure is moderate compared to keeping
    the original documents in RAM.
    """
    out: set[Tuple[str, ...]] = set()
    for text in texts:
        for gram in ngrams(text, n=n):
            out.add(gram)
    return out
