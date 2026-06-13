"""Tests for the cross-benchmark agreement comparison utility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _agreement_receipt_dict(benchmark: str, kappa: float, n_traces: int,
                            shortcut_rate: float = 0.1, run_id: str = "run-x"):
    return {
        "schema_version": "v1",
        "kind": "agreement",
        "run_id": f"{run_id}-{benchmark}",
        "timestamp_utc": "2025-01-01T00:00:00+00:00",
        "git_commit": "deadbeef",
        "model": "stub",
        "judge_model": "stub-judge",
        "judge_prompt_version": "MATH_RUBRIC_V1",
        "benchmark": benchmark,
        "split": "test",
        "n_problems": n_traces,
        "n_traces": n_traces,
        "rm_threshold": None,
        "pairs": [
            {
                "pair_name": "verifier_vs_judge",
                "cohens_kappa": kappa,
                "kappa_ci_95": [max(kappa - 0.05, -1.0), min(kappa + 0.05, 1.0)],
                "agreement_rate": (kappa + 1) / 2,
                "confusion": {"tp": 1, "tn": 1, "fp": 0, "fn": 0, "n": 2, "dropped_uncertain": 0},
                "per_difficulty": {},
            }
        ],
        "shortcut_bucket": {
            "total_verifier_accepted": 10,
            "judge_flagged": int(round(shortcut_rate * 10)),
            "shortcut_rate": shortcut_rate,
            "by_reason": {},
            "examples_path": "shortcuts.jsonl",
        },
        "wall_clock_seconds": 1.0,
        "hardware": {"cpu_only": True},
        "notes": {},
    }


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_render_two_benchmarks(tmp_path):
    from scripts.agreement_compare import main

    gsm8k = _write(tmp_path, "gsm8k.json",
                   _agreement_receipt_dict("gsm8k", 0.74, 800, 0.085))
    math = _write(tmp_path, "math.json",
                  _agreement_receipt_dict("math", 0.41, 200, 0.22))
    out = tmp_path / "cross.md"
    rc = main([
        "--receipt", str(gsm8k),
        "--receipt", str(math),
        "--output", str(out),
    ])
    assert rc == 0
    text = out.read_text()
    assert "Judge Reliability Across Benchmarks" in text
    assert "| gsm8k |" in text
    assert "| math |" in text
    assert "0.740" in text
    assert "0.410" in text
    assert "8.5%" in text
    assert "22.0%" in text
    assert "numeric equivalence" in text
    assert "symbolic equivalence" in text


def test_render_three_benchmarks_includes_humaneval(tmp_path):
    from scripts.agreement_compare import main

    paths = []
    for benchmark, kappa, n, rate in (
        ("gsm8k", 0.74, 800, 0.085),
        ("math", 0.41, 200, 0.22),
        ("humaneval", 0.62, 400, 0.14),
    ):
        paths.append(_write(tmp_path, f"{benchmark}.json",
                            _agreement_receipt_dict(benchmark, kappa, n, rate)))
    out = tmp_path / "cross.md"
    rc = main([
        "--receipt", str(paths[0]),
        "--receipt", str(paths[1]),
        "--receipt", str(paths[2]),
        "--output", str(out),
    ])
    assert rc == 0
    text = out.read_text()
    assert "| humaneval |" in text
    assert "sandbox execution" in text


def test_single_receipt_rejected(tmp_path):
    from scripts.agreement_compare import main

    only = _write(tmp_path, "only.json",
                  _agreement_receipt_dict("gsm8k", 0.7, 10))
    out = tmp_path / "cross.md"
    with pytest.raises(ValueError):
        main(["--receipt", str(only), "--output", str(out)])
