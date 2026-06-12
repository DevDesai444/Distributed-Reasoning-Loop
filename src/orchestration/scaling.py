"""
Scaling-efficiency utilities for distributed training runs.

The training launcher uses these helpers to record per-step throughput and to
turn a single-GPU baseline into a speedup / efficiency number. The math here is
deliberately simple so the unit tests can lock the contract: any future change
to how we report scaling has to update both the helper and the assertions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Dict, Iterable, List, Optional


@dataclass
class StepRecord:
    """Per-step timing record. Wall-clock in seconds, token count if known."""

    step_index: int
    wall_clock_s: float
    tokens: int = 0
    samples: int = 0


@dataclass
class ThroughputSummary:
    """Aggregated throughput numbers from a sequence of `StepRecord`s."""

    n_steps: int
    total_wall_clock_s: float
    tokens_per_sec: float
    samples_per_sec: float
    mean_step_ms: float
    p95_step_ms: float
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        out = {
            "n_steps": float(self.n_steps),
            "total_wall_clock_s": self.total_wall_clock_s,
            "tokens_per_sec": self.tokens_per_sec,
            "samples_per_sec": self.samples_per_sec,
            "mean_step_ms": self.mean_step_ms,
            "p95_step_ms": self.p95_step_ms,
        }
        out.update(self.extras)
        return out


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile. Returns 0.0 on empty input."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower_idx = int(math.floor(rank))
    upper_idx = int(math.ceil(rank))
    if lower_idx == upper_idx:
        return float(sorted_values[lower_idx])
    fraction = rank - lower_idx
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    return float(lower + (upper - lower) * fraction)


def summarize_records(records: Iterable[StepRecord]) -> ThroughputSummary:
    """Collapse a sequence of step records into a throughput summary."""
    record_list = list(records)
    if not record_list:
        return ThroughputSummary(
            n_steps=0,
            total_wall_clock_s=0.0,
            tokens_per_sec=0.0,
            samples_per_sec=0.0,
            mean_step_ms=0.0,
            p95_step_ms=0.0,
        )

    total_wall = sum(record.wall_clock_s for record in record_list)
    total_tokens = sum(record.tokens for record in record_list)
    total_samples = sum(record.samples for record in record_list)
    step_ms_list = [record.wall_clock_s * 1000.0 for record in record_list]

    tokens_per_sec = total_tokens / total_wall if total_wall > 0 else 0.0
    samples_per_sec = total_samples / total_wall if total_wall > 0 else 0.0

    return ThroughputSummary(
        n_steps=len(record_list),
        total_wall_clock_s=total_wall,
        tokens_per_sec=tokens_per_sec,
        samples_per_sec=samples_per_sec,
        mean_step_ms=float(mean(step_ms_list)),
        p95_step_ms=_percentile(step_ms_list, 95.0),
    )


def measure_throughput(
    stage_fn: Callable[[int], Optional[Dict[str, float]]],
    n_steps: int = 100,
) -> ThroughputSummary:
    """
    Drive `stage_fn` for `n_steps` iterations and aggregate the timings.

    `stage_fn(step_index)` should perform one training step and may optionally
    return a dict with `tokens` and `samples` keys so token/sample throughput
    gets surfaced. Steps that return `None` are still timed.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    records: List[StepRecord] = []
    for step_index in range(n_steps):
        start = time.perf_counter()
        result = stage_fn(step_index)
        elapsed = time.perf_counter() - start
        tokens = int(result.get("tokens", 0)) if result else 0
        samples = int(result.get("samples", 0)) if result else 0
        records.append(
            StepRecord(
                step_index=step_index,
                wall_clock_s=elapsed,
                tokens=tokens,
                samples=samples,
            )
        )

    return summarize_records(records)


def scaling_efficiency(
    single_gpu_throughput: float,
    multi_gpu_throughput: float,
    world_size: int,
) -> Dict[str, float]:
    """
    Compute speedup, efficiency_pct, and ideal_speedup for a multi-GPU run.

    Both throughputs are expected to be in the same unit (samples/sec or
    tokens/sec — pick one and stick with it). `world_size` is the number of
    ranks the multi-GPU run used. Ideal speedup is world_size; efficiency is
    speedup divided by ideal, in percent.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if single_gpu_throughput <= 0:
        return {
            "speedup": 0.0,
            "efficiency_pct": 0.0,
            "ideal_speedup": float(world_size),
            "single_gpu_throughput": float(single_gpu_throughput),
            "multi_gpu_throughput": float(multi_gpu_throughput),
            "world_size": float(world_size),
        }

    speedup = multi_gpu_throughput / single_gpu_throughput
    ideal = float(world_size)
    efficiency_pct = (speedup / ideal) * 100.0 if ideal > 0 else 0.0

    return {
        "speedup": float(speedup),
        "efficiency_pct": float(efficiency_pct),
        "ideal_speedup": ideal,
        "single_gpu_throughput": float(single_gpu_throughput),
        "multi_gpu_throughput": float(multi_gpu_throughput),
        "world_size": float(world_size),
    }


def build_scaling_summary(
    summary: ThroughputSummary,
    world_size: int,
    single_gpu_throughput: Optional[float] = None,
    metric: str = "samples_per_sec",
) -> Dict[str, float]:
    """Combine a throughput summary with an optional baseline into one dict."""
    if metric not in {"samples_per_sec", "tokens_per_sec"}:
        raise ValueError(
            f"metric must be 'samples_per_sec' or 'tokens_per_sec', got {metric!r}"
        )

    out: Dict[str, float] = {
        "metric": metric,
        "world_size": float(world_size),
        **summary.to_dict(),
    }

    if single_gpu_throughput is not None:
        multi = summary.tokens_per_sec if metric == "tokens_per_sec" else summary.samples_per_sec
        out.update(scaling_efficiency(single_gpu_throughput, multi, world_size))

    return out
