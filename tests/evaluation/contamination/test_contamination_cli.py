"""End-to-end test for the contamination CLI on a synthetic sample corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.contamination_eval import main as contamination_main


SAMPLE_CORPUS = [
    {"benchmark": "gsm8k", "split": "train", "id": "t1",
     "problem": "Alice has 12 apples and Bob has 7 apples. How many do they have in total together combined?"},
    {"benchmark": "gsm8k", "split": "train", "id": "t2",
     "problem": "A train leaves the station at 3 pm going north at 60 miles per hour for two hours straight."},
    {"benchmark": "gsm8k", "split": "test", "id": "e1",
     # Heavy overlap with t1 by construction.
     "problem": "Alice has 12 apples and Bob has 7 apples. How many do they have in total together combined now?"},
    {"benchmark": "gsm8k", "split": "test", "id": "e2",
     "problem": "Carol baked 9 muffins and ate 2. How many remain on the plate after she finished eating breakfast?"},
]


def _write_sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.jsonl"
    with path.open("w") as fh:
        for row in SAMPLE_CORPUS:
            fh.write(json.dumps(row) + "\n")
    return path


def test_cli_emits_json_and_markdown(tmp_path):
    sample = _write_sample(tmp_path)
    out = tmp_path / "out"
    rc = contamination_main([
        "--benchmark", "gsm8k",
        "--output-dir", str(out),
        "--sample-corpus", str(sample),
        "--force-sample-corpus",
        "--n", "5",
        "--high-threshold", "2",
    ])
    assert rc == 0
    assert (out / "contamination_report.json").exists()
    assert (out / "report.md").exists()
    payload = json.loads((out / "contamination_report.json").read_text())
    assert payload["schema_version"] == "contamination/v1"
    assert payload["meta"]["data_source"] == "sample-fallback"
    # Two passes: train-vs-test and test-vs-train.
    assert len(payload["reports"]) == 2
    test_in_train = next(r for r in payload["reports"] if r["needle_name"] == "gsm8k/test")
    # We engineered e1 to overlap heavily with t1.
    assert test_in_train["needle_problems_with_any_overlap"] >= 1
    assert test_in_train["top_matches"][0]["needle_id"] == "e1"


def test_cli_with_external_corpus(tmp_path):
    sample = _write_sample(tmp_path)
    external = tmp_path / "ext.jsonl"
    with external.open("w") as fh:
        fh.write(json.dumps({"id": "ext1",
                             "text": "Carol baked 9 muffins and ate 2. How many remain on the plate after"}) + "\n")
    out = tmp_path / "out"
    rc = contamination_main([
        "--benchmark", "gsm8k",
        "--output-dir", str(out),
        "--sample-corpus", str(sample),
        "--force-sample-corpus",
        "--external", str(external),
        "--n", "5", "--high-threshold", "2",
    ])
    assert rc == 0
    payload = json.loads((out / "contamination_report.json").read_text())
    names = [(r["haystack_name"], r["needle_name"]) for r in payload["reports"]]
    assert any("external:" in h for h, _ in names)


def test_cli_with_synthetic_pairs(tmp_path):
    sample = _write_sample(tmp_path)
    pairs = tmp_path / "pairs.jsonl"
    with pairs.open("w") as fh:
        fh.write(json.dumps({
            "pair": {
                "problem_id": "pair_0",
                "chosen": "Carol baked 9 muffins and ate 2. How many remain on the plate after she finished",
                "rejected": "irrelevant text token token token token token token token token token token token",
            }
        }) + "\n")
    out = tmp_path / "out"
    rc = contamination_main([
        "--benchmark", "gsm8k",
        "--output-dir", str(out),
        "--sample-corpus", str(sample),
        "--force-sample-corpus",
        "--synthetic-pairs", str(pairs),
        "--n", "5", "--high-threshold", "2",
    ])
    assert rc == 0
    payload = json.loads((out / "contamination_report.json").read_text())
    names = [(r["haystack_name"], r["needle_name"]) for r in payload["reports"]]
    assert any("synthetic_pairs:" in h for h, _ in names)


def test_cli_errors_with_no_data_source(tmp_path):
    """If neither cache nor loader nor sample corpus works, we must fail clearly.

    We invoke with a benchmark the loader won't reach and no sample corpus, and
    expect a RuntimeError (not a silent zero-report). MATH is a reasonable
    target because its loader requires network access we don't have here.
    """
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        contamination_main([
            "--benchmark", "math",
            "--output-dir", str(out),
        ])
