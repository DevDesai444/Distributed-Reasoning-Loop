#!/usr/bin/env python3
"""
CLI for the shortcut process reward model.

Three subcommands:

  build-dataset    convert shortcuts.jsonl + correct_samples.jsonl into
                   train/eval JSONL splits.
  fit              train the shortcut PRM on those splits.
  score            apply a trained PRM to a JSONL of traces and emit
                   per-trace scores.

This module is wired into ``main.py`` as ``train shortcut-prm <sub>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------- build-dataset


def _cmd_build_dataset(rest: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="train shortcut-prm build-dataset")
    parser.add_argument("--shortcuts", type=str, required=True)
    parser.add_argument("--clean", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    args = parser.parse_args(rest)

    from training.shortcut_prm_dataset import build_shortcut_prm_dataset

    stats = build_shortcut_prm_dataset(
        args.shortcuts,
        args.clean,
        args.output_dir,
        balance=not args.no_balance,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
    )
    print(json.dumps(stats.to_dict(), indent=2))
    return 0


# --------------------------------------------------------------------- fit


def _cmd_fit(rest: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="train shortcut-prm fit")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--smoke-max-steps", type=int, default=3)
    args = parser.parse_args(rest)

    from training.shortcut_prm_trainer import TrainConfig, train_shortcut_prm

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        if not args.cpu_only:
            raise SystemExit("--dataset-dir is required unless --cpu-only is set")
        dataset_dir = _materialize_smoke_dataset(Path(args.output_dir))

    config = TrainConfig(
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        cpu_only=args.cpu_only,
        smoke_max_steps=args.smoke_max_steps,
    )
    result = train_shortcut_prm(config)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _materialize_smoke_dataset(output_dir: Path) -> str:
    """Generate the smoke dataset + JSONL splits on the fly for --cpu-only."""
    from training.shortcut_prm_dataset import build_shortcut_prm_dataset

    smoke_root = output_dir / "smoke_inputs"
    samples_dir = smoke_root / "samples"
    dataset_dir = smoke_root / "dataset"

    # Lazy import to avoid a hard dep when --dataset-dir is supplied.
    from scripts.shortcut_smoke_data import write_smoke_dataset

    paths = write_smoke_dataset(samples_dir, n=50)
    build_shortcut_prm_dataset(
        paths["shortcuts"],
        paths["clean"],
        dataset_dir,
        balance=True,
        seed=17,
        eval_fraction=0.1,
    )
    return str(dataset_dir)


# ------------------------------------------------------------------- score


def _cmd_score(rest: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="train shortcut-prm score")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a JSONL of traces. Each row is read with the "
                             "key --text-key (default 'text').")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(rest)

    from training.shortcut_prm import ShortcutPRM

    prm = ShortcutPRM.load(args.model_dir)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    texts: List[str] = []
    with open(input_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(args.text_key) or row.get("reasoning") or row.get("response") or ""
            rows.append(row)
            texts.append(text)

    scores: List[float] = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        scores.extend(prm.score_batch(batch))

    with open(output_path, "w") as handle:
        for row, score in zip(rows, scores):
            out = dict(row)
            out["shortcut_probability"] = float(score)
            handle.write(json.dumps(out) + "\n")

    print(f"scored {len(scores)} traces -> {output_path}")
    return 0


# ---------------------------------------------------------------- dispatch


SUBCOMMANDS = {
    "build-dataset": _cmd_build_dataset,
    "fit": _cmd_fit,
    "score": _cmd_score,
}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv) if argv is not None else []
    if not argv:
        print(
            "usage: train shortcut-prm <build-dataset|fit|score> [...]",
            file=sys.stderr,
        )
        return 2
    sub = argv[0]
    rest = argv[1:]
    handler = SUBCOMMANDS.get(sub)
    if handler is None:
        print(f"unknown shortcut-prm subcommand: {sub!r}", file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
