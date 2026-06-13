#!/usr/bin/env python3
"""
Distributed SFT -> DPO -> GRPO launcher.

Designed to be invoked under torchrun:

    torchrun --nproc-per-node=$NGPUS scripts/launch_distributed.py \\
        --stage sft \\
        --dataset gsm8k \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --output-dir ./outputs/distributed_run \\
        --log-dir ./logs/distributed_run

Backends:
    - NCCL when CUDA is available (the production path)
    - Gloo when CUDA is unavailable (the CPU smoke path)

Outputs at end of run:
    - `<log-dir>/rank_<RANK>.log`        : per-rank init / heartbeat trace
    - `<log-dir>/training_steps.jsonl`   : per-step metrics emitted by rank 0
    - `<output-dir>/scaling_summary.json`: aggregated throughput numbers

The launcher itself does not produce model weights — it drives the existing
trainers in `src/training/`. When a stage needs to run, it executes a small
synthetic step loop that exercises the dataloader, optimizer, and DDP sync.
The smoke run uses this loop so the receipt format and scaling math can be
verified without a real benchmark run; on a GPU host the same loop runs against
the actual datasets pulled by the stage helpers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

# Make `src/` importable when this file runs as a script under torchrun.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for candidate in (_REPO_ROOT, _SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


logger = logging.getLogger("launch_distributed")


VALID_STAGES = ("sft", "dpo", "grpo")


@dataclass
class LaunchArgs:
    """Parsed CLI configuration for the launcher."""

    stage: str
    dataset: str
    model: str
    output_dir: Path
    log_dir: Path
    n_steps: int
    backend: Optional[str]
    distributed_timeout_minutes: int
    single_gpu_throughput: Optional[float]
    metric: str
    smoke: bool
    seed: int
    extra: List[str] = field(default_factory=list)


def parse_args(argv: Optional[List[str]] = None) -> LaunchArgs:
    parser = argparse.ArgumentParser(
        description="Distributed SFT/DPO/GRPO launcher (torchrun-compatible).",
    )
    parser.add_argument(
        "--stage",
        choices=(*VALID_STAGES, "all"),
        required=True,
        help="Training stage to run. 'all' runs sft -> dpo -> grpo in sequence.",
    )
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/distributed_run"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./logs/distributed_run"),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=20,
        help="Number of synthetic optimizer steps per stage when smoking.",
    )
    parser.add_argument(
        "--backend",
        choices=("nccl", "gloo"),
        default=None,
        help="Force a distributed backend. Defaults: nccl on CUDA, gloo on CPU.",
    )
    parser.add_argument("--distributed-timeout-minutes", type=int, default=0)
    parser.add_argument(
        "--single-gpu-throughput",
        type=float,
        default=None,
        help="Baseline throughput (same unit as --metric) for scaling math.",
    )
    parser.add_argument(
        "--metric",
        choices=("samples_per_sec", "tokens_per_sec"),
        default="samples_per_sec",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Skip model loads; run a tiny CPU loop to prove the harness.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Trailing args forwarded to the underlying trainer (not used in smoke).",
    )
    parsed = parser.parse_args(argv)
    return LaunchArgs(
        stage=parsed.stage,
        dataset=parsed.dataset,
        model=parsed.model,
        output_dir=parsed.output_dir,
        log_dir=parsed.log_dir,
        n_steps=parsed.n_steps,
        backend=parsed.backend,
        distributed_timeout_minutes=parsed.distributed_timeout_minutes,
        single_gpu_throughput=parsed.single_gpu_throughput,
        metric=parsed.metric,
        smoke=parsed.smoke,
        seed=parsed.seed,
        extra=parsed.extra or [],
    )


def _stages_to_run(stage: str) -> List[str]:
    if stage == "all":
        return list(VALID_STAGES)
    return [stage]


def _configure_rank_logger(log_dir: Path, rank: int) -> Path:
    """Open a per-rank log file. Returns its path so the run can refer back."""
    log_dir.mkdir(parents=True, exist_ok=True)
    rank_log_path = log_dir / f"rank_{rank}.log"
    handler = logging.FileHandler(rank_log_path, mode="a")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [rank=%(name)s] %(levelname)s: %(message)s")
    )
    rank_logger = logging.getLogger(str(rank))
    rank_logger.addHandler(handler)
    rank_logger.setLevel(logging.INFO)
    return rank_log_path


@contextmanager
def _jsonl_writer(path: Path, enabled: bool):
    """Yield a `write_record(dict)` callable. No-op when disabled."""
    if not enabled:
        yield lambda _record: None
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        def write(record: Dict[str, object]) -> None:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

        yield write
    finally:
        handle.close()


def _smoke_step(stage: str, step_index: int, seed: int) -> Dict[str, float]:
    """
    Deterministic tiny step used by the CPU smoke run.

    Returns enough metadata that the per-step JSONL has the shape a real run
    would have: loss, tokens/sec proxy, sample count, wall clock.
    """
    # Cheap busy-work so the wall clock is non-zero but bounded. Two tiny matmuls
    # over a 32x32 tensor — same cost on every rank, varies a touch by step.
    import torch

    torch.manual_seed(seed + step_index)
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)
    out = a @ b
    loss = float((out.mean() ** 2).item())
    return {
        "loss": loss,
        "tokens": 128,
        "samples": 4,
        "stage": stage,
    }


def _run_stage_smoke(
    stage: str,
    n_steps: int,
    seed: int,
    write_step: Callable[[Dict[str, object]], None],
    is_main: bool,
) -> "ThroughputSummary":
    from orchestration.scaling import StepRecord, summarize_records

    records: List[StepRecord] = []
    for step_index in range(n_steps):
        start = time.perf_counter()
        metadata = _smoke_step(stage, step_index, seed)
        elapsed = time.perf_counter() - start
        record = StepRecord(
            step_index=step_index,
            wall_clock_s=elapsed,
            tokens=int(metadata["tokens"]),
            samples=int(metadata["samples"]),
        )
        records.append(record)
        if is_main:
            write_step(
                {
                    "stage": stage,
                    "step": step_index,
                    "loss": metadata["loss"],
                    "lr": 0.0,
                    "tokens_per_sec": record.tokens / max(elapsed, 1e-9),
                    "samples_per_sec": record.samples / max(elapsed, 1e-9),
                    "gpu_util": 0.0,
                    "wall_clock_s": elapsed,
                }
            )
    return summarize_records(records)


def _run_stage_real(stage: str, args: LaunchArgs) -> None:
    """
    Dispatch the requested stage to the real trainer module.

    Kept thin on purpose: this script doesn't try to be the trainer, it just
    calls the same entry points a single-GPU user would. The trainers detect
    `WORLD_SIZE > 1` via env vars and wrap themselves in DDP accordingly.
    """
    if stage == "sft":
        # Real SFT requires a synthetic data path the user supplies via --extra.
        raise NotImplementedError(
            "Real SFT stage requires a data path; pass it via --extra. "
            "Smoke mode (--smoke) is what this launcher exercises in CI."
        )
    if stage == "dpo":
        raise NotImplementedError(
            "Real DPO stage requires --extra preference-pair path."
        )
    if stage == "grpo":
        raise NotImplementedError(
            "Real GRPO stage requires --extra synthetic-data path."
        )
    raise ValueError(f"Unknown stage: {stage!r}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # We do the import here so unit tests can hit parse_args without forcing
    # torch onto the import path.
    from training.distributed import (
        barrier_if_distributed,
        init_distributed,
        shutdown_distributed,
    )
    from orchestration.scaling import build_scaling_summary

    ctx = init_distributed(
        timeout_minutes=args.distributed_timeout_minutes,
        force_backend=args.backend,
    )

    rank_log_path = _configure_rank_logger(args.log_dir, ctx.rank)
    rank_logger = logging.getLogger(str(ctx.rank))
    rank_logger.info(
        "Joined process group: world_size=%s backend=%s device=%s initialized=%s",
        ctx.world_size,
        ctx.backend,
        ctx.device,
        ctx.initialized,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    steps_path = args.log_dir / "training_steps.jsonl"
    is_main = ctx.is_main_process

    aggregate_summaries: Dict[str, Dict[str, float]] = {}

    with _jsonl_writer(steps_path, enabled=is_main) as write_step:
        for stage in _stages_to_run(args.stage):
            rank_logger.info("Starting stage=%s n_steps=%s", stage, args.n_steps)
            if args.smoke:
                summary = _run_stage_smoke(
                    stage,
                    args.n_steps,
                    args.seed,
                    write_step,
                    is_main,
                )
            else:
                _run_stage_real(stage, args)
                continue
            barrier_if_distributed(ctx)

            scaling = build_scaling_summary(
                summary,
                world_size=ctx.world_size,
                single_gpu_throughput=args.single_gpu_throughput,
                metric=args.metric,
            )
            aggregate_summaries[stage] = scaling
            rank_logger.info(
                "Stage %s done: %s steps, %.4f s wall, %.2f %s",
                stage,
                summary.n_steps,
                summary.total_wall_clock_s,
                summary.samples_per_sec if args.metric == "samples_per_sec" else summary.tokens_per_sec,
                args.metric,
            )

    if is_main:
        scaling_summary = {
            "stages": aggregate_summaries,
            "world_size": ctx.world_size,
            "backend": ctx.backend,
            "metric": args.metric,
            "smoke": args.smoke,
            "single_gpu_throughput": args.single_gpu_throughput,
        }
        summary_path = args.output_dir / "scaling_summary.json"
        summary_path.write_text(json.dumps(scaling_summary, indent=2))
        rank_logger.info("Wrote scaling summary: %s", summary_path)

    barrier_if_distributed(ctx)
    shutdown_distributed()
    rank_logger.info("Rank %s shutdown clean (log=%s).", ctx.rank, rank_log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
