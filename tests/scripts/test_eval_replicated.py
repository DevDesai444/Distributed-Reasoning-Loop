"""Tests for the seed-replicated eval driver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eval_replicated  # noqa: E402


def _fake_runner(metrics_by_seed):
    """Build a SeedRunner that returns canned SeedResults per seed."""

    def run(seed: int) -> eval_replicated.SeedResult:
        record = metrics_by_seed[seed]
        return eval_replicated.SeedResult(
            seed=seed,
            pass_at_1=record["pass_at_1"],
            pass_at_4=record.get("pass_at_4"),
            pass_at_8=record.get("pass_at_8"),
            n_problems=record.get("n_problems", 100),
        )

    return run


def test_aggregate_metric_mean_and_std():
    out = eval_replicated.aggregate_metric("pass_at_1", [0.5, 0.6, 0.7])
    assert out.n == 3
    assert abs(out.mean - 0.6) < 1e-9
    # Sample std for [0.5,0.6,0.7] is 0.1
    assert abs(out.std - 0.1) < 1e-9
    assert out.ci_low <= out.mean <= out.ci_high


def test_aggregate_metric_single_value_has_zero_std():
    out = eval_replicated.aggregate_metric("pass_at_1", [0.42])
    assert out.std == 0.0
    assert out.mean == 0.42
    # Degenerate CI: both endpoints equal the only value.
    assert out.ci_low == 0.42
    assert out.ci_high == 0.42


def test_aggregate_metric_empty():
    out = eval_replicated.aggregate_metric("pass_at_1", [])
    assert out.n == 0
    assert out.mean == 0.0


def test_bootstrap_ci_brackets_mean():
    values = [0.5, 0.55, 0.6, 0.65, 0.7]
    low, high = eval_replicated.bootstrap_ci(values, n_resamples=200, seed=7)
    mean = sum(values) / len(values)
    assert low <= mean <= high


def test_aggregate_results_pass_at_4_when_all_seeds_have_it():
    results = [
        eval_replicated.SeedResult(seed=0, pass_at_1=0.5, pass_at_4=0.7),
        eval_replicated.SeedResult(seed=1, pass_at_1=0.6, pass_at_4=0.8),
    ]
    aggregate = eval_replicated.aggregate_results(results)
    assert "pass_at_1" in aggregate
    assert "pass_at_4" in aggregate
    assert "pass_at_8" not in aggregate


def test_aggregate_results_skips_partial_metrics():
    results = [
        eval_replicated.SeedResult(seed=0, pass_at_1=0.5, pass_at_4=0.7),
        # Second seed missing pass_at_4 — we should not invent a number for it.
        eval_replicated.SeedResult(seed=1, pass_at_1=0.6, pass_at_4=None),
    ]
    aggregate = eval_replicated.aggregate_results(results)
    assert "pass_at_4" not in aggregate


def test_run_replicated_with_fake_runner(tmp_path):
    metrics = {
        0: {"pass_at_1": 0.5, "pass_at_4": 0.7, "pass_at_8": 0.85},
        1: {"pass_at_1": 0.55, "pass_at_4": 0.72, "pass_at_8": 0.88},
        2: {"pass_at_1": 0.6, "pass_at_4": 0.74, "pass_at_8": 0.9},
    }
    payload = eval_replicated.run_replicated([0, 1, 2], _fake_runner(metrics))
    assert payload["n_seeds"] == 3
    assert len(payload["per_seed"]) == 3
    assert "pass_at_1" in payload["aggregate"]
    assert payload["aggregate"]["pass_at_1"]["n"] == 3


def test_main_writes_output_file(tmp_path, monkeypatch):
    """Run the CLI entry point with a stubbed runner factory."""
    output = tmp_path / "eval_replicated.json"

    def fake_factory(**kwargs):
        return _fake_runner(
            {
                0: {"pass_at_1": 0.5},
                1: {"pass_at_1": 0.6},
            }
        )

    monkeypatch.setattr(eval_replicated, "_default_runner_factory", fake_factory)
    rc = eval_replicated.main(
        [
            "--model",
            "fake-model",
            "--benchmark",
            "gsm8k",
            "--seeds",
            "0",
            "1",
            "--n-problems",
            "5",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()
    payload = json.loads(output.read_text())
    assert payload["n_seeds"] == 2
    assert "pass_at_1" in payload["aggregate"]
