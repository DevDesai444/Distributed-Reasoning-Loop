"""Smoke test: run contamination on real GSM8K (if cache available) or sample fallback.

This test serves as the guarantee that we ship a real contamination report path
that runs end-to-end. If the local HuggingFace cache doesn't have GSM8K, the
test falls back to the bundled sample corpus and still verifies the report
file is well-formed.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.contamination_eval import main as contamination_main

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_FALLBACK = REPO_ROOT / "data" / "contamination_sample_corpus.jsonl"


def test_real_run_produces_valid_report(tmp_path):
    out = tmp_path / "out"
    argv = [
        "--benchmark", "gsm8k",
        "--output-dir", str(out),
        # Cap haystack/needles so the test stays fast even on the full corpus.
        "--max-haystack", "200",
        "--max-needles", "200",
    ]
    if SAMPLE_FALLBACK.exists():
        argv += ["--sample-corpus", str(SAMPLE_FALLBACK)]
    rc = contamination_main(argv)
    assert rc == 0
    report_path = out / "contamination_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["schema_version"] == "contamination/v1"
    assert payload["meta"]["benchmark"] == "gsm8k"
    assert isinstance(payload["reports"], list)
    assert len(payload["reports"]) >= 1
    for r in payload["reports"]:
        # Sanity: every report must include the fields downstream tooling reads.
        for key in ("haystack_name", "needle_name", "n",
                    "haystack_total_problems", "needle_total_problems",
                    "needle_problems_with_any_overlap",
                    "needle_problems_with_high_overlap",
                    "overlap_rate_any", "overlap_rate_high",
                    "top_matches"):
            assert key in r
