"""Tests for the distributed launcher script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import launch_distributed  # noqa: E402


def test_parse_args_minimal():
    args = launch_distributed.parse_args(
        ["--stage", "sft"],
    )
    assert args.stage == "sft"
    assert args.dataset == "gsm8k"
    assert args.smoke is False


def test_parse_args_invalid_stage():
    with pytest.raises(SystemExit):
        launch_distributed.parse_args(["--stage", "ppo"])


def test_stages_to_run_all_expands():
    assert launch_distributed._stages_to_run("all") == list(launch_distributed.VALID_STAGES)


def test_stages_to_run_single():
    assert launch_distributed._stages_to_run("dpo") == ["dpo"]


def test_smoke_step_returns_loss_and_token_counts():
    metadata = launch_distributed._smoke_step("sft", step_index=0, seed=42)
    assert metadata["stage"] == "sft"
    assert metadata["tokens"] == 128
    assert metadata["samples"] == 4
    assert isinstance(metadata["loss"], float)


def test_launcher_smoke_end_to_end(tmp_path, monkeypatch):
    """Run the launcher's main() with smoke mode and assert artifact layout."""
    output_dir = tmp_path / "outputs"
    log_dir = tmp_path / "logs"

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    rc = launch_distributed.main(
        [
            "--stage",
            "sft",
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(log_dir),
            "--backend",
            "gloo",
            "--smoke",
            "--n-steps",
            "3",
            "--single-gpu-throughput",
            "1.0",
        ]
    )
    assert rc == 0

    scaling_path = output_dir / "scaling_summary.json"
    assert scaling_path.exists()
    payload = json.loads(scaling_path.read_text())
    assert payload["world_size"] == 1
    assert payload["backend"] == "gloo"
    assert payload["smoke"] is True
    assert "sft" in payload["stages"]
    assert payload["stages"]["sft"]["n_steps"] == 3.0

    rank_log = log_dir / "rank_0.log"
    assert rank_log.exists()
    assert "Joined process group" in rank_log.read_text()

    steps_log = log_dir / "training_steps.jsonl"
    assert steps_log.exists()
    lines = steps_log.read_text().splitlines()
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert record["stage"] == "sft"
    assert "loss" in record
    assert "tokens_per_sec" in record


def test_launcher_smoke_all_stages(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    rc = launch_distributed.main(
        [
            "--stage",
            "all",
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(log_dir),
            "--backend",
            "gloo",
            "--smoke",
            "--n-steps",
            "2",
        ]
    )
    assert rc == 0
    payload = json.loads((output_dir / "scaling_summary.json").read_text())
    assert set(payload["stages"]) == set(launch_distributed.VALID_STAGES)
