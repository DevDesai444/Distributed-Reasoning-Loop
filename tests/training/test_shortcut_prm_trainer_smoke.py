"""End-to-end CPU smoke for the shortcut-PRM trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_smoke_training_runs(tmp_path):
    from scripts.shortcut_smoke_data import write_smoke_dataset
    from training.shortcut_prm_dataset import build_shortcut_prm_dataset
    from training.shortcut_prm_trainer import TrainConfig, train_shortcut_prm

    samples = tmp_path / "samples"
    write_smoke_dataset(samples, n=20)
    dataset_dir = tmp_path / "dataset"
    build_shortcut_prm_dataset(
        samples / "shortcuts.jsonl",
        samples / "correct_samples.jsonl",
        dataset_dir,
        balance=True,
        seed=17,
    )

    out_dir = tmp_path / "smoke_model"
    config = TrainConfig(
        dataset_dir=str(dataset_dir),
        output_dir=str(out_dir),
        n_epochs=1,
        smoke_max_steps=3,
        batch_size=4,
        cpu_only=True,
    )
    result = train_shortcut_prm(config)

    assert result.final_metrics is not None
    assert 0.0 <= result.final_metrics.eval_accuracy <= 1.0
    assert 0.0 <= result.final_metrics.eval_f1 <= 1.0
    assert 0.0 <= result.final_metrics.eval_auc <= 1.0

    log_path = Path(result.train_log_path)
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert rows
    for row in rows:
        for key in ("epoch", "train_loss", "eval_loss", "eval_accuracy", "eval_f1", "eval_auc"):
            assert key in row

    assert (Path(result.output_dir) / "train_config.json").exists()
