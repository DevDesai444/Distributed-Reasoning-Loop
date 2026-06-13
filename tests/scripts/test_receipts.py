"""
Tests for receipt schemas and the receipt collector.

Two layers:

  1. Schema round-trip tests. Build each receipt from a synthetic dict
     fixture, serialize to JSON, read back, assert fields match. These
     run in <100 ms.

  2. End-to-end smoke tests. Drive `scripts/collect_receipts.py` in
     smoke mode and verify the three receipt files appear and pass
     validate(). Also exercises crash-safety via the hidden
     `--fail-stage` flag.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Make scripts/ importable regardless of how pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scripts.receipt_schemas import (  # noqa: E402
    RECEIPT_SCHEMA_VERSION,
    AgreementReceipt,
    RobustnessReceipt,
    SyntheticReceipt,
    load_receipt,
)


# -- fixtures ---------------------------------------------------------------


def _agreement_fixture() -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "kind": "agreement",
        "run_id": "agreement_test_001",
        "timestamp_utc": "2026-06-12T00:00:00+00:00",
        "git_commit": "deadbeef",
        "model": "test/policy",
        "judge_model": "test/judge",
        "judge_prompt_version": "MATH_RUBRIC_V1",
        "benchmark": "gsm8k",
        "split": "test",
        "n_problems": 200,
        "n_traces": 800,
        "rm_threshold": 0.5,
        "pairs": [
            {
                "pair_name": "verifier_vs_judge",
                "cohens_kappa": 0.74,
                "kappa_ci_95": [0.69, 0.79],
                "agreement_rate": 0.86,
                "confusion": {"tp": 400, "tn": 250, "fp": 80, "fn": 70, "n": 800},
                "per_difficulty": {
                    "low": {"kappa": 0.81, "agreement": 0.9, "n": 300},
                    "mid": {"kappa": 0.72, "agreement": 0.85, "n": 300},
                    "high": {"kappa": 0.65, "agreement": 0.78, "n": 200},
                },
            },
        ],
        "shortcut_bucket": {
            "total_verifier_accepted": 612,
            "judge_flagged": 52,
            "shortcut_rate": 0.085,
            "by_reason": {"LUCKY_GUESS": 8, "WRONG_METHOD": 13},
            "examples_path": "agreement_run/shortcuts.jsonl",
        },
        "wall_clock_seconds": 1234.5,
        "hardware": {"gpu": "T4", "vram_gb": 16},
        "notes": {},
    }


def _robustness_fixture() -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "kind": "robustness",
        "run_id": "robustness_test_001",
        "timestamp_utc": "2026-06-12T00:00:00+00:00",
        "git_commit": "deadbeef",
        "model": "test/policy",
        "benchmark": "gsm8k",
        "split": "test",
        "n_problems": 200,
        "n_samples_per_problem": 4,
        "seed": 0,
        "overall_robustness": 0.785,
        "per_perturbation": [
            {
                "name": "number_swap",
                "rewrites_label": True,
                "applicable_rate": 0.61,
                "matched_baseline_pass_at_1": 0.55,
                "perturbed_pass_at_1": 0.42,
                "retention": 0.764,
                "retention_ci_95": [0.71, 0.81],
            },
        ],
        "wall_clock_seconds": 980.0,
        "hardware": {"gpu": "T4", "vram_gb": 16},
        "notes": {},
    }


def _synthetic_fixture() -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "kind": "synthetic",
        "run_id": "synthetic_test_001",
        "timestamp_utc": "2026-06-12T00:00:00+00:00",
        "git_commit": "deadbeef",
        "model": "test/generator",
        "dataset": "gsm8k",
        "split": "train",
        "target_pairs": 30000,
        "stats": {
            "total_problems_attempted": 7473,
            "total_generations": 89124,
            "verifier_accepts": 44612,
            "verifier_rejects": 44512,
            "verifier_accept_rate": 0.5,
            "pair_candidates": 38400,
            "unique_pairs_after_dedup": 30245,
            "dedup_ratio": 0.788,
            "per_backend": {"vllm": {"generations": 89124, "accepts": 44612}},
            "quality_score_distribution": {"p50": 0.78, "p90": 0.91, "p99": 0.97},
        },
        "wall_clock_seconds": 14200.0,
        "hardware": {"gpu": "T4", "vram_gb": 16},
        "notes": {},
    }


# -- round-trip tests -------------------------------------------------------


def _roundtrip(receipt, tmp_path: Path, name: str) -> Dict[str, Any]:
    out = tmp_path / name
    receipt.to_json(out)
    return json.loads(out.read_text())


def test_agreement_schema_roundtrip(tmp_path: Path) -> None:
    fixture = _agreement_fixture()
    receipt = AgreementReceipt.from_dict(fixture)
    receipt.validate()
    written = _roundtrip(receipt, tmp_path, "agreement_v1.json")
    assert written["kind"] == "agreement"
    assert written["pairs"][0]["pair_name"] == "verifier_vs_judge"
    assert written["pairs"][0]["cohens_kappa"] == pytest.approx(0.74)
    assert written["shortcut_bucket"]["shortcut_rate"] == pytest.approx(0.085)
    reloaded = AgreementReceipt.from_file(tmp_path / "agreement_v1.json")
    reloaded.validate()
    assert reloaded.n_problems == 200


def test_robustness_schema_roundtrip(tmp_path: Path) -> None:
    fixture = _robustness_fixture()
    receipt = RobustnessReceipt.from_dict(fixture)
    receipt.validate()
    written = _roundtrip(receipt, tmp_path, "robustness_v1.json")
    assert written["kind"] == "robustness"
    assert written["overall_robustness"] == pytest.approx(0.785)
    assert written["per_perturbation"][0]["name"] == "number_swap"
    reloaded = RobustnessReceipt.from_file(tmp_path / "robustness_v1.json")
    reloaded.validate()
    assert reloaded.n_samples_per_problem == 4


def test_synthetic_schema_roundtrip(tmp_path: Path) -> None:
    fixture = _synthetic_fixture()
    receipt = SyntheticReceipt.from_dict(fixture)
    receipt.validate()
    written = _roundtrip(receipt, tmp_path, "synthetic_v1.json")
    assert written["kind"] == "synthetic"
    assert written["stats"]["unique_pairs_after_dedup"] == 30245
    reloaded = SyntheticReceipt.from_file(tmp_path / "synthetic_v1.json")
    reloaded.validate()
    assert reloaded.target_pairs == 30000


def test_receipt_schema_version_constant() -> None:
    """Guard against silent schema bumps. If the bump is intentional,
    bump this assertion in the same commit so the intent is reviewable."""
    assert RECEIPT_SCHEMA_VERSION == "v1"


def test_load_receipt_dispatches_by_kind(tmp_path: Path) -> None:
    fixture = _agreement_fixture()
    path = tmp_path / "agreement_v1.json"
    path.write_text(json.dumps(fixture))
    receipt = load_receipt(path)
    assert isinstance(receipt, AgreementReceipt)


def test_validate_rejects_wrong_kind(tmp_path: Path) -> None:
    bad = _agreement_fixture()
    bad["kind"] = "agreement"
    bad["schema_version"] = "v999"
    r = AgreementReceipt.from_dict(bad)
    with pytest.raises(ValueError):
        r.validate()


# -- end-to-end smoke tests -------------------------------------------------


def _run_collector(*extra_args: str, output_dir: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "collect_receipts.py"),
        "--mode", "smoke",
        "--output-dir", str(output_dir),
        *extra_args,
    ]
    return subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)


def test_collect_receipts_smoke_mode(tmp_path: Path) -> None:
    out = tmp_path / "receipts_smoke"
    res = _run_collector("--stage", "all", output_dir=out)
    assert res.returncode == 0, f"collector failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    receipts = out / "receipts"
    assert (receipts / "synthetic_v1.json").exists()
    assert (receipts / "agreement_v1.json").exists()
    assert (receipts / "robustness_v1.json").exists()

    # Every receipt round-trips through the loader without raising.
    for path in receipts.glob("*.json"):
        receipt = load_receipt(path)
        receipt.validate()

    # Validate stage re-reads them in place.
    res_validate = _run_collector("--stage", "validate", output_dir=out)
    assert res_validate.returncode == 0, res_validate.stderr


def test_collect_receipts_crash_safe(tmp_path: Path) -> None:
    """If one stage fails, the other completed stages still have receipts."""
    out = tmp_path / "receipts_crash"
    res = _run_collector(
        "--stage", "all",
        "--fail-stage", "agreement",
        output_dir=out,
    )
    # Collector returns non-zero overall, but synthetic + robustness still wrote.
    assert res.returncode == 1, res.stderr
    receipts = out / "receipts"
    assert (receipts / "synthetic_v1.json").exists(), \
        f"synthetic receipt missing: stderr=\n{res.stderr}"
    assert (receipts / "robustness_v1.json").exists()
    assert not (receipts / "agreement_v1.json").exists()

    summary = json.loads((out / "collector_summary.json").read_text())
    assert "agreement" in summary["errors"]
    assert summary["receipts_written"].get("synthetic", "").endswith("synthetic_v1.json")
