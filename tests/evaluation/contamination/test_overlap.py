"""Tests for the overlap analyzer."""

from __future__ import annotations

from src.evaluation.contamination.overlap import analyze_overlap, report_to_dict


def _doc(doc_id: str, text: str) -> dict:
    return {"id": doc_id, "text": text}


def test_identical_documents_yield_high_overlap():
    shared = "the quick brown fox jumps over the lazy dog and runs fast"
    haystack = [_doc("h1", shared)]
    needles = [_doc("n1", shared)]
    report = analyze_overlap(haystack, needles, n=5, high_threshold=3)
    assert report.needle_total_problems == 1
    assert report.needle_problems_with_any_overlap == 1
    assert report.needle_problems_with_high_overlap == 1
    assert report.top_matches[0].needle_id == "n1"
    assert report.top_matches[0].overlap_count >= 3


def test_disjoint_corpora_yield_zero_overlap():
    haystack = [_doc("h1", "alpha beta gamma delta epsilon zeta eta theta iota kappa")]
    needles = [_doc("n1", "one two three four five six seven eight nine ten")]
    report = analyze_overlap(haystack, needles, n=5)
    assert report.needle_problems_with_any_overlap == 0
    assert report.overlap_rate_any == 0.0
    assert report.top_matches == []


def test_partial_overlap_counts_correctly():
    haystack = [_doc("h1", "the quick brown fox jumps over the lazy dog")]
    needles = [_doc("n1", "the quick brown fox runs away")]
    report = analyze_overlap(haystack, needles, n=3, high_threshold=2)
    # n-grams in needle: (the,quick,brown), (quick,brown,fox), (brown,fox,runs), (fox,runs,away)
    # First two are in haystack.
    assert report.needle_problems_with_any_overlap == 1
    assert report.needle_problems_with_high_overlap == 1
    assert report.top_matches[0].overlap_count == 2


def test_high_threshold_filters_low_overlap():
    haystack = [_doc("h1", "the quick brown fox")]
    needles = [_doc("n1", "the quick brown cat sat on the mat")]
    report = analyze_overlap(haystack, needles, n=3, high_threshold=3)
    # Only one matching 3-gram. Hits any-overlap, misses high-overlap.
    assert report.needle_problems_with_any_overlap == 1
    assert report.needle_problems_with_high_overlap == 0


def test_top_k_limits_match_list():
    haystack = [_doc("h1", "alpha beta gamma delta epsilon zeta")]
    needles = [_doc(f"n{i}", "alpha beta gamma delta epsilon zeta") for i in range(15)]
    report = analyze_overlap(haystack, needles, n=3, top_k=5)
    assert len(report.top_matches) == 5


def test_empty_needles_yields_zero_rate_without_div_zero():
    haystack = [_doc("h1", "anything goes here at all")]
    report = analyze_overlap(haystack, [], n=3)
    assert report.overlap_rate_any == 0.0
    assert report.needle_total_problems == 0


def test_top_matches_sorted_by_overlap_count():
    haystack = [_doc("h1", "the cat sat on the mat and looked at the sun")]
    needles = [
        _doc("low", "the cat sat in space"),
        _doc("high", "the cat sat on the mat and looked at the sun in space"),
    ]
    report = analyze_overlap(haystack, needles, n=3, top_k=2)
    assert report.top_matches[0].needle_id == "high"
    assert report.top_matches[1].needle_id == "low"


def test_report_to_dict_round_trips():
    haystack = [_doc("h1", "the quick brown fox jumps over the lazy dog")]
    needles = [_doc("n1", "the quick brown fox jumps over the lazy dog")]
    report = analyze_overlap(haystack, needles, n=5)
    payload = report_to_dict(report)
    assert payload["needle_total_problems"] == 1
    assert payload["top_matches"][0]["needle_id"] == "n1"
    # Tuples must serialise to lists for JSON.
    assert isinstance(payload["top_matches"][0]["overlapping_ngrams"][0], list)


def test_accepts_problem_field_as_alternative_to_text():
    haystack = [{"id": "h1", "problem": "alpha beta gamma delta epsilon"}]
    needles = [{"id": "n1", "problem": "alpha beta gamma delta epsilon"}]
    report = analyze_overlap(haystack, needles, n=3)
    assert report.needle_problems_with_any_overlap == 1
