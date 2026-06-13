"""Tests for the shortcut-PRM dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.shortcut_prm_dataset import build_shortcut_prm_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_inputs(tmp_path: Path, n_short: int = 6, n_clean: int = 12) -> tuple[Path, Path]:
    short_rows = [
        {
            "pair_id": f"short_{i}",
            "response_excerpt": f"shortcut trace {i} answer 42",
            "reason_code": "INCOMPLETE_REASONING",
        }
        for i in range(n_short)
    ]
    clean_rows = [
        {
            "pair_id": f"clean_{i}",
            "reasoning": f"step 1 then step 2 then answer for trace {i}",
        }
        for i in range(n_clean)
    ]
    short_path = tmp_path / "shortcuts.jsonl"
    clean_path = tmp_path / "correct_samples.jsonl"
    _write_jsonl(short_path, short_rows)
    _write_jsonl(clean_path, clean_rows)
    return short_path, clean_path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_dataset_builder_balanced_split(tmp_path):
    short_path, clean_path = _make_inputs(tmp_path)
    out_dir = tmp_path / "out"

    stats = build_shortcut_prm_dataset(short_path, clean_path, out_dir)

    train_rows = _read_jsonl(out_dir / "train.jsonl")
    eval_rows = _read_jsonl(out_dir / "eval.jsonl")

    # balance=True downsamples majority to 6, so 12 total, 10/2 train/eval at 0.1
    assert stats.train_n + stats.eval_n == 12
    assert stats.eval_n == max(1, round(12 * 0.1))

    short_train = sum(1 for r in train_rows if r["label"] == 1)
    clean_train = sum(1 for r in train_rows if r["label"] == 0)
    short_eval = sum(1 for r in eval_rows if r["label"] == 1)
    clean_eval = sum(1 for r in eval_rows if r["label"] == 0)
    assert short_train + short_eval == 6
    assert clean_train + clean_eval == 6


def test_dataset_builder_preserves_source_id(tmp_path):
    short_path, clean_path = _make_inputs(tmp_path, n_short=4, n_clean=4)
    out_dir = tmp_path / "out"
    build_shortcut_prm_dataset(short_path, clean_path, out_dir)

    rows = _read_jsonl(out_dir / "train.jsonl") + _read_jsonl(out_dir / "eval.jsonl")
    for row in rows:
        assert "source_id" in row and row["source_id"]


def test_dataset_builder_train_eval_disjoint(tmp_path):
    short_path, clean_path = _make_inputs(tmp_path, n_short=10, n_clean=10)
    out_dir = tmp_path / "out"
    build_shortcut_prm_dataset(short_path, clean_path, out_dir)

    train_ids = {r["source_id"] for r in _read_jsonl(out_dir / "train.jsonl")}
    eval_ids = {r["source_id"] for r in _read_jsonl(out_dir / "eval.jsonl")}
    assert not (train_ids & eval_ids)


def test_dataset_builder_deterministic_under_fixed_seed(tmp_path):
    short_path, clean_path = _make_inputs(tmp_path, n_short=8, n_clean=8)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    build_shortcut_prm_dataset(short_path, clean_path, out_a, seed=99)
    build_shortcut_prm_dataset(short_path, clean_path, out_b, seed=99)

    train_a = _read_jsonl(out_a / "train.jsonl")
    train_b = _read_jsonl(out_b / "train.jsonl")
    assert [r["source_id"] for r in train_a] == [r["source_id"] for r in train_b]


def test_dataset_filters_overlapping_ids(tmp_path):
    short_rows = [{"pair_id": "shared_0", "response_excerpt": "shortcut text"}]
    clean_rows = [
        {"pair_id": "shared_0", "reasoning": "should be filtered"},
        {"pair_id": "uniq_1", "reasoning": "kept"},
    ]
    short_path = tmp_path / "shortcuts.jsonl"
    clean_path = tmp_path / "correct_samples.jsonl"
    _write_jsonl(short_path, short_rows)
    _write_jsonl(clean_path, clean_rows)
    out_dir = tmp_path / "out"
    build_shortcut_prm_dataset(short_path, clean_path, out_dir, balance=False)

    rows = _read_jsonl(out_dir / "train.jsonl") + _read_jsonl(out_dir / "eval.jsonl")
    clean_ids = {r["source_id"] for r in rows if r["label"] == 0}
    assert "shared_0" not in clean_ids
    assert "uniq_1" in clean_ids
