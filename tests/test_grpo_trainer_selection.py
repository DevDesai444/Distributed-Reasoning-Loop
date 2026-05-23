"""
Tests for GRPO checkpoint-selection behavior.
"""

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
