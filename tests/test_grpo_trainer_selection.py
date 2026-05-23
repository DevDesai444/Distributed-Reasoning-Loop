"""
Tests for GRPO checkpoint-selection behavior.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.grpo_trainer import GRPOConfig, ReasoningGRPOTrainer


def test_grpo_rejects_unknown_checkpoint_metric():
    with pytest.raises(ValueError, match="Unsupported best_checkpoint_metric"):
        ReasoningGRPOTrainer(
            GRPOConfig(best_checkpoint_metric="unknown_metric")
        )


def test_grpo_selection_metric_uses_configured_metric():
    trainer = ReasoningGRPOTrainer(
        GRPOConfig(best_checkpoint_metric="mean_reward")
    )

    value = trainer._selection_metric_value(
        {
            "pass_at_1": 0.6,
            "mean_reward": 0.25,
        }
    )

    assert value == 0.25


def test_grpo_best_checkpoint_manifest_records_selection_metadata(tmp_path, monkeypatch):
    trainer = ReasoningGRPOTrainer(
        GRPOConfig(
            best_checkpoint_metric="pass_at_1",
            output_dir=str(tmp_path / "grpo_output"),
            save_best_checkpoint=True,
        )
    )

    monkeypatch.setattr(
        trainer,
        "save",
        lambda path, merge_lora=False: Path(path).mkdir(parents=True, exist_ok=True),
    )

    should_stop = trainer._maybe_update_best_checkpoint(
        step=42,
        metrics={"pass_at_1": 0.73, "mean_reward": 0.2},
    )

    manifest = json.loads(
        (tmp_path / "grpo_output" / "best_checkpoint" / "stage_manifest.json").read_text()
    )

    assert should_stop is False
    assert manifest["selection_metric"] == "pass_at_1"
    assert manifest["selection_score"] == 0.73
    assert manifest["selection_step"] == 42
