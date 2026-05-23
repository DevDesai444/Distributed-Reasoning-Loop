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
