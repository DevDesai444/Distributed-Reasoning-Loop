"""
Dataset builder for the shortcut process reward model.

Reads the two JSONL artifacts emitted by an agreement run:

- ``shortcuts.jsonl`` — traces the verifier accepted but the judge flagged
  as shortcuts (label 1).
- ``correct_samples.jsonl`` — traces the verifier accepted and the judge
  did not flag (label 0). Any pair_id appearing in ``shortcuts.jsonl`` is
  filtered out so the two splits are disjoint.

Produces a balanced (optional), deterministic 90/10 train/eval split as
``train.jsonl`` and ``eval.jsonl`` in the output directory.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)


LABEL_SHORTCUT = 1
LABEL_CLEAN = 0
DEFAULT_SEED = 17


@dataclass
class DatasetStats:
    """Summary of a built shortcut-PRM dataset."""

    train_n: int
    eval_n: int
    class_balance: dict[str, int]
    source_files: dict[str, str]
    seed: int
    balanced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_n": self.train_n,
            "eval_n": self.eval_n,
            "class_balance": self.class_balance,
            "source_files": self.source_files,
            "seed": self.seed,
            "balanced": self.balanced,
        }


# ----------------------------------------------------------------- helpers


def _read_jsonl(path: Path) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _shortcut_text(record: dict[str, Any]) -> Optional[str]:
    """Extract the trace text from a shortcuts.jsonl row."""
    for key in ("response_excerpt", "response", "reasoning", "trace"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _clean_text(record: dict[str, Any]) -> Optional[str]:
    """Extract the trace text from a correct_samples.jsonl row."""
    for key in ("reasoning", "response", "response_excerpt", "trace"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _source_id(record: dict[str, Any], fallback: str) -> str:
    for key in ("pair_id", "id", "trace_id", "problem_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            count += 1
    return count


# ----------------------------------------------------------------- builder


def build_shortcut_prm_dataset(
    shortcuts_jsonl: str | Path,
    clean_jsonl: str | Path,
    output_dir: str | Path,
    *,
    balance: bool = True,
    seed: int = DEFAULT_SEED,
    eval_fraction: float = 0.1,
) -> DatasetStats:
    """
    Build train/eval JSONL files for the shortcut PRM.

    Each output row is::

        {"text": ..., "label": 0|1, "source_id": ...}
    """
    shortcuts_path = Path(shortcuts_jsonl)
    clean_path = Path(clean_jsonl)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shortcut_records = _read_jsonl(shortcuts_path)
    clean_records = _read_jsonl(clean_path)

    shortcut_rows: List[dict[str, Any]] = []
    shortcut_ids: set[str] = set()
    for idx, raw in enumerate(shortcut_records):
        text = _shortcut_text(raw)
        if text is None:
            continue
        sid = _source_id(raw, fallback=f"shortcut_{idx}")
        shortcut_ids.add(sid)
        shortcut_rows.append(
            {"text": text, "label": LABEL_SHORTCUT, "source_id": sid}
        )

    clean_rows: List[dict[str, Any]] = []
    for idx, raw in enumerate(clean_records):
        sid = _source_id(raw, fallback=f"clean_{idx}")
        if sid in shortcut_ids:
            continue
        text = _clean_text(raw)
        if text is None:
            continue
        clean_rows.append(
            {"text": text, "label": LABEL_CLEAN, "source_id": sid}
        )

    if not shortcut_rows:
        raise ValueError(f"No shortcut rows extracted from {shortcuts_path}")
    if not clean_rows:
        raise ValueError(f"No clean rows extracted from {clean_path}")

    rng = random.Random(seed)
    rng.shuffle(shortcut_rows)
    rng.shuffle(clean_rows)

    if balance:
        n = min(len(shortcut_rows), len(clean_rows))
        shortcut_rows = shortcut_rows[:n]
        clean_rows = clean_rows[:n]

    combined: List[dict[str, Any]] = shortcut_rows + clean_rows
    rng.shuffle(combined)

    if eval_fraction <= 0 or eval_fraction >= 1:
        raise ValueError("eval_fraction must be in (0, 1)")
    eval_n = max(1, int(round(len(combined) * eval_fraction)))
    eval_n = min(eval_n, len(combined) - 1)
    eval_rows = combined[:eval_n]
    train_rows = combined[eval_n:]

    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    train_written = _write_jsonl(train_rows, train_path)
    eval_written = _write_jsonl(eval_rows, eval_path)

    class_balance = {
        "train_shortcut": sum(1 for row in train_rows if row["label"] == LABEL_SHORTCUT),
        "train_clean": sum(1 for row in train_rows if row["label"] == LABEL_CLEAN),
        "eval_shortcut": sum(1 for row in eval_rows if row["label"] == LABEL_SHORTCUT),
        "eval_clean": sum(1 for row in eval_rows if row["label"] == LABEL_CLEAN),
    }
    stats = DatasetStats(
        train_n=train_written,
        eval_n=eval_written,
        class_balance=class_balance,
        source_files={
            "shortcuts": str(shortcuts_path),
            "clean": str(clean_path),
        },
        seed=seed,
        balanced=balance,
    )
    with open(out_dir / "dataset_stats.json", "w") as handle:
        json.dump(stats.to_dict(), handle, indent=2)

    logger.info(
        "Built shortcut-PRM dataset: train=%s eval=%s balance=%s",
        train_written,
        eval_written,
        class_balance,
    )
    return stats
