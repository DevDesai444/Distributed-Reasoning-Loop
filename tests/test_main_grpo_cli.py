"""
Tests for the standalone GRPO CLI entrypoint in main.py.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as drl_main


def test_cmd_train_grpo_forwards_checkpoint_selection_args(monkeypatch):
    captured = {}

    def fake_maybe_launch_grpo_distributed(training_args, requested_num_gpus="auto"):
        captured["distributed_args"] = training_args
        captured["requested_num_gpus"] = requested_num_gpus
        return False

    def fake_train_grpo_from_synthetic_data(**kwargs):
        captured["train_kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "src.training.grpo_trainer",
        types.SimpleNamespace(
            maybe_launch_grpo_distributed=fake_maybe_launch_grpo_distributed,
            train_grpo_from_synthetic_data=fake_train_grpo_from_synthetic_data,
        ),
    )

    class Args:
        remaining = [
            "--data-path", "pairs.jsonl",
            "--heldout-dataset", "gsm8k",
            "--heldout-split", "test",
            "--heldout-eval-size", "12",
            "--save-best-checkpoint",
            "--best-checkpoint-metric", "pass_at_1",
            "--min-eval-improvement", "0.002",
            "--early-stop-patience", "4",
        ]

    drl_main.cmd_train_grpo(Args())

    assert "--heldout-dataset" in captured["distributed_args"]
    assert "--save-best-checkpoint" in captured["distributed_args"]
    assert captured["requested_num_gpus"] == "auto"
    assert captured["train_kwargs"]["heldout_dataset"] == "gsm8k"
    assert captured["train_kwargs"]["heldout_eval_size"] == 12
    assert captured["train_kwargs"]["save_best_checkpoint"] is True
    assert captured["train_kwargs"]["early_stop_patience"] == 4
