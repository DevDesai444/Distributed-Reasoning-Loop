#!/usr/bin/env python3
"""
Launch the Layer-0 pre-training run.

Usage (single node, 4 GPU)::

    torchrun --nproc_per_node=4 scripts/pretrain.py \
        --data-dir ./data/pretraining \
        --output-dir ./outputs/pretraining \
        --total-steps 10000 \
        --precision bf16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow ``python scripts/pretrain.py`` to find the src/ package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.training.pretraining import (
    PreTrainer,
    PreTrainerConfig,
)
from src.training.pretraining.data import PretrainingDataConfig
from src.training.pretraining.model import TitanModelConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DRL Layer-0 pre-training")
    p.add_argument("--data-dir", default="./data/pretraining")
    p.add_argument("--file-glob", default="shard-*.jsonl")
    p.add_argument("--output-dir", default="./outputs/pretraining")
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("--ffn-hidden", type=int, default=2816)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-learning-rate", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--total-steps", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--precision", choices=("fp32", "bf16", "fp8"), default="bf16")
    p.add_argument("--peak-tflops", type=float, default=312.0)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    model_cfg = TitanModelConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_hidden=args.ffn_hidden,
    )
    data_cfg = PretrainingDataConfig(
        data_dir=args.data_dir,
        file_glob=args.file_glob,
        seq_len=args.seq_len,
    )
    trainer_cfg = PreTrainerConfig(
        model=model_cfg,
        data=data_cfg,
        precision=args.precision,
        output_dir=args.output_dir,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        peak_tflops=args.peak_tflops,
    )

    PreTrainer(trainer_cfg).train()


if __name__ == "__main__":
    main()
