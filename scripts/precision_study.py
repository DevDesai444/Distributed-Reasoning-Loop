#!/usr/bin/env python3
"""
FP32 vs BF16 vs FP8 study.

Runs a short, identical training window in each precision and reports
throughput, peak memory, gradient-norm stability, and final-eval delta. Output
is a JSON file consumed by reports and dashboards.

Usage (single GPU, short)::

    python scripts/precision_study.py \
        --precisions fp32,bf16 \
        --steps 100 \
        --output ./outputs/precision_study.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from src.training.precision import (
    PrecisionConfig,
    PrecisionPolicy,
    SUPPORTED_PRECISIONS,
)
from src.training.pretraining.data import (
    PretrainingDataConfig,
    StreamingTokenizedDataset,
    synthetic_token_shards,
)
from src.training.pretraining.model import TitanModel, TitanModelConfig

logger = logging.getLogger(__name__)


def _peak_memory_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _make_model(seq_len: int, vocab: int) -> TitanModel:
    config = TitanModelConfig(
        vocab_size=vocab,
        max_seq_len=seq_len,
        n_layers=4,
        d_model=256,
        n_heads=8,
        n_kv_heads=4,
        ffn_hidden=512,
    )
    return TitanModel(config)


def _run_window(
    precision: str,
    *,
    seq_len: int,
    vocab: int,
    steps: int,
    batch_size: int,
    data_dir: str,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    model = _make_model(seq_len=seq_len, vocab=vocab).to(device)
    policy = PrecisionPolicy(PrecisionConfig(precision=precision))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    dataset_config = PretrainingDataConfig(
        data_dir=data_dir, seq_len=seq_len, file_glob="shard-*.jsonl"
    )
    loader = torch.utils.data.DataLoader(
        StreamingTokenizedDataset(dataset_config),
        batch_size=batch_size,
        drop_last=True,
    )

    _reset_peak_memory()
    grad_norms: List[float] = []
    losses: List[float] = []
    tokens_processed = 0

    model.train()
    start = time.perf_counter()
    iterator = iter(loader)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with policy.autocast():
            _, loss = model(input_ids, labels=labels)
        policy.backward(loss)
        record = policy.step(optimizer, model.parameters(), step=step)
        optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().item()))
        grad_norms.append(record.post_clip_norm)
        tokens_processed += input_ids.numel()

    elapsed = time.perf_counter() - start
    tokens_per_second = tokens_processed / max(elapsed, 1.0e-6)
    grad_std = statistics.stdev(grad_norms) if len(grad_norms) > 1 else 0.0
    return {
        "precision": precision,
        "tokens_per_second": tokens_per_second,
        "peak_memory_mib": _peak_memory_mib(),
        "final_loss": losses[-1] if losses else float("nan"),
        "mean_loss": sum(losses) / len(losses) if losses else float("nan"),
        "grad_norm_mean": sum(grad_norms) / len(grad_norms) if grad_norms else float("nan"),
        "grad_norm_std": grad_std,
        "overflow_steps": sum(
            1 for record in policy.grad_norm_history if record.overflow
        ),
        "steps": steps,
        "device": str(device),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FP32 vs BF16 vs FP8 precision study")
    p.add_argument(
        "--precisions",
        default="fp32,bf16",
        help="Comma-separated precisions to sweep.",
    )
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--data-dir", default="./outputs/precision_study/shards")
    p.add_argument("--output", default="./outputs/precision_study/report.json")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    requested = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]
    unknown = [p for p in requested if p not in SUPPORTED_PRECISIONS]
    if unknown:
        raise SystemExit(f"Unknown precisions requested: {unknown}")

    os.makedirs(args.data_dir, exist_ok=True)
    synthetic_token_shards(
        args.data_dir,
        num_shards=4,
        tokens_per_shard=args.steps * args.seq_len * args.batch_size,
        vocab_size=args.vocab_size,
        seed=11,
    )

    results: List[Dict[str, float]] = []
    for precision in requested:
        logger.info("Running precision sweep entry: %s", precision)
        results.append(
            _run_window(
                precision,
                seq_len=args.seq_len,
                vocab=args.vocab_size,
                steps=args.steps,
                batch_size=args.batch_size,
                data_dir=args.data_dir,
            )
        )

    if results:
        baseline = results[0]
        for entry in results:
            entry["throughput_speedup_vs_first"] = (
                entry["tokens_per_second"] / baseline["tokens_per_second"]
                if baseline["tokens_per_second"]
                else 0.0
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump({"runs": results}, fh, indent=2)
    logger.info("Wrote precision study report to %s", args.output)
    for entry in results:
        logger.info(
            "%s tokens/s=%.0f peak_mib=%.1f mean_loss=%.4f grad_std=%.4f overflows=%d",
            entry["precision"],
            entry["tokens_per_second"],
            entry["peak_memory_mib"],
            entry["mean_loss"],
            entry["grad_norm_std"],
            entry["overflow_steps"],
        )


if __name__ == "__main__":
    main()
