"""
Tests for explicit run artifact management.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_artifacts import RunArtifacts


def test_run_artifacts_create_manifest_and_latest_pointer(tmp_path):
    outputs_dir = tmp_path / "outputs"
    manager = RunArtifacts(
        root_output_dir=str(outputs_dir),
        dataset="gsm8k",
        training_method="best",
        config={"seed": 42},
        run_name="artifact-test",
    )

    manager.record_metric("gsm8k_accuracy", 0.71)
    manager.record_artifact(
        stage="evaluation",
        name="gsm8k_results",
        path=manager.run_dir / "gsm8k_results.json",
        metadata={"accuracy": 0.71},
    )
    manager.finalize("completed", summary={"accuracy": 0.71})

    manifest = json.loads((manager.run_dir / "run_manifest.json").read_text())
    latest = json.loads((outputs_dir / "latest_run.json").read_text())
    summary = json.loads((manager.run_dir / "run_summary.json").read_text())

    assert manifest["run_id"] == "artifact-test"
    assert manifest["status"] == "completed"
    assert manifest["metrics"]["gsm8k_accuracy"] == 0.71
    assert latest["run_dir"] == str(manager.run_dir.resolve())
    assert summary["accuracy"] == 0.71


def test_run_artifacts_can_attach_to_flat_output_directory(tmp_path):
    outputs_dir = tmp_path / "benchmark_results"
    manager = RunArtifacts(
        root_output_dir=str(outputs_dir),
        dataset="gsm8k",
        training_method="evaluation",
        config={"subset_size": 100},
        nested=False,
    )

    manager.record_metric("gsm8k_accuracy", 0.71)
    manager.finalize("completed", summary={"valid_run": True})

    manifest = json.loads((outputs_dir / "run_manifest.json").read_text())
    latest = json.loads((outputs_dir / "latest_run.json").read_text())
    summary = json.loads((outputs_dir / "run_summary.json").read_text())

    assert manifest["run_dir"] == str(outputs_dir.resolve())
    assert latest["run_dir"] == str(outputs_dir.resolve())
    assert summary["valid_run"] is True


def test_run_artifacts_can_compare_against_previous_run(tmp_path):
    outputs_dir = tmp_path / "outputs"
    first = RunArtifacts(
        root_output_dir=str(outputs_dir),
        dataset="gsm8k",
        training_method="best",
        run_name="run-a",
    )
    first.finalize(
        "completed",
        summary={"evaluation": {"gsm8k": {"accuracy": 0.67}}},
    )

    second = RunArtifacts(
        root_output_dir=str(outputs_dir),
        dataset="gsm8k",
        training_method="best",
        run_name="run-b",
    )
    comparison = second.compare_summary_to_previous(
        {"evaluation": {"gsm8k": {"accuracy": 0.71}}}
    )

    assert comparison is not None
    assert comparison["metrics"]["evaluation.gsm8k.accuracy"]["previous"] == 0.67
    assert comparison["metrics"]["evaluation.gsm8k.accuracy"]["current"] == 0.71
