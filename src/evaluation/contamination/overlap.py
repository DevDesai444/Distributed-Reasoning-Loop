"""
Overlap analysis between a haystack corpus and a needle corpus.

The asymmetric framing (haystack vs needles) is deliberate: we report, for each
needle problem, how many of its n-grams appear anywhere in the haystack. That
catches the typical contamination case where a test problem appears verbatim
or near-verbatim inside the training corpus. Running it both directions
(train-in-test and test-in-train) catches asymmetric leaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

from src.evaluation.contamination.ngrams import ngram_set, ngrams


@dataclass(frozen=True)
class ContaminationMatch:
    """A single needle document with n-grams found in the haystack."""

    needle_id: str
    needle_text: str
    overlapping_ngrams: List[Tuple[str, ...]]
    overlap_count: int


@dataclass(frozen=True)
class ContaminationReport:
    """Summary of a single haystack-vs-needle pass.

    ``overlap_rate_any`` is the fraction of needle documents that share at
    least one n-gram with the haystack. ``overlap_rate_high`` raises the bar
    to ``high_threshold`` matches (default 3), which filters incidental
    one-phrase coincidences.
    """

    haystack_name: str
    needle_name: str
    n: int
    high_threshold: int
    haystack_total_problems: int
    needle_total_problems: int
    needle_problems_with_any_overlap: int
    needle_problems_with_high_overlap: int
    overlap_rate_any: float
    overlap_rate_high: float
    top_matches: List[ContaminationMatch] = field(default_factory=list)


def _problem_text(p: dict) -> str:
    """Pull the text field, accepting either ``text`` or ``problem``."""
    if "text" in p:
        return p["text"]
    if "problem" in p:
        return p["problem"]
    raise KeyError("problem dict must have a 'text' or 'problem' key")


def analyze_overlap(
    haystack: List[dict],
    needles: List[dict],
    n: int = 13,
    high_threshold: int = 3,
    top_k: int = 10,
    haystack_name: str = "haystack",
    needle_name: str = "needles",
    max_example_ngrams: int = 5,
) -> ContaminationReport:
    """Count n-gram overlap of each needle against the union of haystack n-grams.

    Implementation note: we build the haystack n-gram set once (O(H) memory)
    and then stream each needle, accumulating only the per-needle match counts
    plus a bounded sample of matching n-grams for the top-k report. We never
    materialise per-needle n-gram lists for the whole corpus.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if high_threshold < 1:
        raise ValueError(f"high_threshold must be >= 1, got {high_threshold}")

    haystack_grams = ngram_set((_problem_text(p) for p in haystack), n=n)

    matches: List[ContaminationMatch] = []
    any_overlap = 0
    high_overlap = 0

    for needle in needles:
        text = _problem_text(needle)
        needle_id = str(needle.get("id", ""))
        overlap_count = 0
        examples: List[Tuple[str, ...]] = []
        for gram in ngrams(text, n=n):
            if gram in haystack_grams:
                overlap_count += 1
                if len(examples) < max_example_ngrams:
                    examples.append(gram)
        if overlap_count > 0:
            any_overlap += 1
            if overlap_count >= high_threshold:
                high_overlap += 1
            matches.append(
                ContaminationMatch(
                    needle_id=needle_id,
                    needle_text=text,
                    overlapping_ngrams=examples,
                    overlap_count=overlap_count,
                )
            )

    matches.sort(key=lambda m: m.overlap_count, reverse=True)
    top_matches = matches[:top_k]

    needle_total = len(needles)
    rate_any = (any_overlap / needle_total) if needle_total else 0.0
    rate_high = (high_overlap / needle_total) if needle_total else 0.0

    return ContaminationReport(
        haystack_name=haystack_name,
        needle_name=needle_name,
        n=n,
        high_threshold=high_threshold,
        haystack_total_problems=len(haystack),
        needle_total_problems=needle_total,
        needle_problems_with_any_overlap=any_overlap,
        needle_problems_with_high_overlap=high_overlap,
        overlap_rate_any=rate_any,
        overlap_rate_high=rate_high,
        top_matches=top_matches,
    )


def report_to_dict(report: ContaminationReport) -> dict:
    """Serialise a report for JSON output. Tuples become lists."""
    return {
        "haystack_name": report.haystack_name,
        "needle_name": report.needle_name,
        "n": report.n,
        "high_threshold": report.high_threshold,
        "haystack_total_problems": report.haystack_total_problems,
        "needle_total_problems": report.needle_total_problems,
        "needle_problems_with_any_overlap": report.needle_problems_with_any_overlap,
        "needle_problems_with_high_overlap": report.needle_problems_with_high_overlap,
        "overlap_rate_any": report.overlap_rate_any,
        "overlap_rate_high": report.overlap_rate_high,
        "top_matches": [
            {
                "needle_id": m.needle_id,
                "needle_text": m.needle_text,
                "overlap_count": m.overlap_count,
                "overlapping_ngrams": [list(g) for g in m.overlapping_ngrams],
            }
            for m in report.top_matches
        ],
    }
