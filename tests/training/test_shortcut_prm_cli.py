"""CLI dispatch tests for `train shortcut-prm`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_build_dataset_subcommand(tmp_path):
    from scripts.shortcut_prm_cli import main as cli_main
    from scripts.shortcut_smoke_data import write_smoke_dataset

    samples = tmp_path / "samples"
    write_smoke_dataset(samples, n=10)
    out_dir = tmp_path / "ds"

    rc = cli_main(
        [
            "build-dataset",
            "--shortcuts", str(samples / "shortcuts.jsonl"),
            "--clean", str(samples / "correct_samples.jsonl"),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "eval.jsonl").exists()
    assert (out_dir / "dataset_stats.json").exists()


def test_fit_subcommand_with_cpu_only(tmp_path):
    from scripts.shortcut_prm_cli import main as cli_main

    out_dir = tmp_path / "smoke"
    rc = cli_main(
        [
            "fit",
            "--cpu-only",
            "--output-dir", str(out_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--smoke-max-steps", "2",
        ]
    )
    assert rc == 0
    assert (out_dir / "train_log.jsonl").exists()
    assert (out_dir / "train_config.json").exists()


def test_unknown_subcommand_returns_2():
    from scripts.shortcut_prm_cli import main as cli_main

    assert cli_main(["bogus"]) == 2
    assert cli_main([]) == 2


def test_score_subcommand_writes_probabilities(tmp_path):
    from scripts.shortcut_prm_cli import main as cli_main

    # build a smoke model first
    out_dir = tmp_path / "smoke"
    cli_main(
        [
            "fit",
            "--cpu-only",
            "--output-dir", str(out_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--smoke-max-steps", "1",
        ]
    )

    # score-only path requires HF tokenizer load — skip if save format didn't
    # preserve a loadable tokenizer (the trivial-tokenizer fallback isn't
    # reloadable by AutoTokenizer)
    try:
        from training.shortcut_prm import ShortcutPRM
        ShortcutPRM.load(out_dir)
    except Exception:
        pytest.skip("smoke model not reloadable via HF AutoTokenizer (trivial tokenizer)")

    input_path = tmp_path / "to_score.jsonl"
    with open(input_path, "w") as handle:
        for i in range(3):
            handle.write(json.dumps({"text": f"trace {i}"}) + "\n")

    output_path = tmp_path / "scored.jsonl"
    rc = cli_main(
        [
            "score",
            "--model-dir", str(out_dir),
            "--input", str(input_path),
            "--output", str(output_path),
        ]
    )
    assert rc == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert all("shortcut_probability" in r for r in rows)
