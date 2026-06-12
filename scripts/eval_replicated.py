#!/usr/bin/env python3
"""
Seed-replicated benchmark evaluation.

The idea is simple: run the same evaluation `n_seeds` times with different
seeds, aggregate pass@k as mean +/- std across seeds, and emit a bootstrap CI
on the mean. Single-seed numbers are easy to overfit to; this helps catch
runs whose lift comes from sampling noise.

Usage:
    python scripts/eval_replicated.py \\
        --model ./outputs/distributed_run/grpo \\
        --benchmark gsm8k \\
        --seeds 0 1 2 \\
        --n-problems 100 \\
        --output ./outputs/distributed_run/eval_replicated.json

The actual per-seed run is delegated to a callback so this module is testable
without loading a model. The default runner shells out to `main.py evaluate`
with a deterministic seed env var.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("eval_replicated")


@dataclass
class SeedResult:
    """Per-seed metrics from one evaluation pass."""

    seed: int
    pass_at_1: float
    pass_at_4: Optional[float] = None
    pass_at_8: Optional[float] = None
    n_problems: int = 0
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass
class AggregateResult:
    """Mean +/- std and bootstrap CI for one metric across seeds."""

    metric: str
    mean: float
    std: float
    n: int
    ci_low: float
    ci_high: float
    ci_level: float


def _safe_std(values: Sequence[float]) -> float:
    """Sample std; returns 0 for n<2 so a single-seed run doesn't crash."""
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI on the mean.

    Degenerate input (empty or single-value) returns (mean, mean). We don't
    try to be clever about small-n; the caller knows how many seeds it ran.
    """
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = random.Random(seed)
    means: List[float] = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))

    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(math.floor(alpha * n_resamples)))
    hi_idx = min(n_resamples - 1, int(math.ceil((1.0 - alpha) * n_resamples)) - 1)
    return float(means[lo_idx]), float(means[hi_idx])


def aggregate_metric(
    metric: str,
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> AggregateResult:
    """Build an AggregateResult for one metric across seeds."""
    if not values:
        return AggregateResult(
            metric=metric,
            mean=0.0,
            std=0.0,
            n=0,
            ci_low=0.0,
            ci_high=0.0,
            ci_level=confidence,
        )
    mean_value = float(statistics.fmean(values))
    std_value = _safe_std(values)
    ci_low, ci_high = bootstrap_ci(
        values,
        confidence=confidence,
        seed=bootstrap_seed,
    )
    return AggregateResult(
        metric=metric,
        mean=mean_value,
        std=std_value,
        n=len(values),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=confidence,
    )


def aggregate_results(
    results: Sequence[SeedResult],
    *,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> Dict[str, AggregateResult]:
    """Aggregate pass@k metrics (and any extras present in every seed) across seeds."""
    if not results:
        return {}

    metric_values: Dict[str, List[float]] = {
        "pass_at_1": [result.pass_at_1 for result in results],
    }
    # pass_at_4 / pass_at_8 only roll up when every seed reported them; partial
    # coverage would give a mean over a smaller-n subset that the metric name
    # would not warn the reader about.
    if all(result.pass_at_4 is not None for result in results):
        metric_values["pass_at_4"] = [float(result.pass_at_4) for result in results]
    if all(result.pass_at_8 is not None for result in results):
        metric_values["pass_at_8"] = [float(result.pass_at_8) for result in results]

    # Same rule for extras: only include keys present on every seed.
    extras_keys = set(results[0].extras)
    for result in results[1:]:
        extras_keys &= set(result.extras)
    for key in sorted(extras_keys):
        metric_values[key] = [float(result.extras[key]) for result in results]

    return {
        metric: aggregate_metric(
            metric,
            values,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        )
        for metric, values in metric_values.items()
    }


def _serialize_aggregate(aggregate: Dict[str, AggregateResult]) -> Dict[str, Dict[str, float]]:
    return {name: asdict(result) for name, result in aggregate.items()}


# --------- per-seed runner --------------------------------------------------

SeedRunner = Callable[[int], SeedResult]


def _default_runner_factory(
    *,
    model: str,
    benchmark: str,
    n_problems: int,
    main_path: Path,
    extra_args: Sequence[str],
) -> SeedRunner:
    """
    Default runner: shell out to `python main.py evaluate ...` with PYTHONHASHSEED
    and EVAL_SEED env vars set per seed, then parse the resulting JSON artifact.

    This is the production path. Tests inject their own runner instead.
    """

    def run(seed: int) -> SeedResult:
        artifact_dir = Path(f"./outputs/eval_seed_{seed}")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(main_path),
            "evaluate",
            "--model",
            model,
            "--benchmark",
            benchmark,
            "--n-problems",
            str(n_problems),
            "--seed",
            str(seed),
            "--output",
            str(artifact_dir / "results.json"),
            *extra_args,
        ]
        env = {"PYTHONHASHSEED": str(seed), "EVAL_SEED": str(seed)}
        logger.info("Running seed=%s: %s", seed, " ".join(cmd))
        subprocess.run(cmd, check=True, env={**dict(__import__("os").environ), **env})

        result_path = artifact_dir / "results.json"
        payload = json.loads(result_path.read_text())
        return SeedResult(
            seed=seed,
            pass_at_1=float(payload.get("pass_at_1", payload.get("accuracy", 0.0))),
            pass_at_4=payload.get("pass_at_4"),
            pass_at_8=payload.get("pass_at_8"),
            n_problems=int(payload.get("n_problems", payload.get("total_problems", n_problems))),
        )

    return run


# --------- top-level orchestration -----------------------------------------

def run_replicated(
    seeds: Iterable[int],
    runner: SeedRunner,
    *,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> Dict[str, object]:
    """
    Drive `runner(seed)` over every seed, aggregate, return a serializable dict.
    """
    per_seed: List[SeedResult] = []
    for seed in seeds:
        result = runner(seed)
        per_seed.append(result)

    aggregate = aggregate_results(
        per_seed,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )

    return {
        "per_seed": [asdict(record) for record in per_seed],
        "aggregate": _serialize_aggregate(aggregate),
        "n_seeds": len(per_seed),
        "confidence_level": confidence,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed-replicated benchmark evaluation with mean +/- std and bootstrap CI.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="gsm8k")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seed values to replicate over.",
    )
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs/eval_replicated.json"),
    )
    parser.add_argument(
        "--main-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "main.py",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to `main.py evaluate`.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    runner = _default_runner_factory(
        model=args.model,
        benchmark=args.benchmark,
        n_problems=args.n_problems,
        main_path=args.main_path,
        extra_args=args.extra or [],
    )

    payload = run_replicated(
        args.seeds,
        runner,
        confidence=args.confidence,
        bootstrap_seed=args.bootstrap_seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote replicated eval to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
