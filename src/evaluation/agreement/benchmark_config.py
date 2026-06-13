"""
Per-benchmark wiring for the three-way agreement runner.

Each benchmark plugs three pieces into the same pipeline:

  * a verifier (math-numeric / math-symbolic / code-execution)
  * a difficulty binner (step count / MATH level / uniform)
  * a judge rubric type (``"math"`` or ``"code"``)

The agreement runner and its tests look up these pieces through
`benchmark_config(name)` so the swap is a single point of truth and
easy to extend.

The helper is import-light: it returns callables and factory
functions rather than instantiating heavy verifiers eagerly. That keeps
``--help`` fast on a fresh CPU box where Docker / sympy may not be
fully wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .difficulty_binner import (
    UNIFORM_BIN,
    bin_for_problem as gsm8k_bin_for_problem,
    uniform_bin,
)
from .math_difficulty_binner import (
    ALL_BINS as MATH_LEVEL_BINS,
    bin_for_problem as math_bin_for_problem,
)


BENCHMARK_NAMES = ("gsm8k", "math", "humaneval")


@dataclass
class BenchmarkConfig:
    """Wiring for one benchmark."""

    name: str
    problem_type: str  # judge rubric key — "math" or "code"
    verifier_factory: Callable[[], Any]
    bin_for_problem: Callable[[Any], str]
    bin_names: Tuple[str, ...]


def _make_gsm8k_verifier() -> Any:
    # GSM8K answers are integers — use the numeric-only specialization.
    from verifier import GSM8KVerifier  # type: ignore

    return GSM8KVerifier()


def _make_math_verifier() -> Any:
    # MATH ships boxed answers; defer to the general symbolic+numeric verifier.
    from verifier import MathVerifier  # type: ignore

    return MathVerifier()


def _make_humaneval_verifier() -> Any:
    """Pick a HumanEval verifier honoring the CPU-no-Docker smoke path.

    The full sandbox (``ExecutionVerifier``) needs Docker. On a host
    without Docker we fall back to ``CodeVerifier`` which uses a
    subprocess-based path good enough for smoke and CI. Real Kaggle
    runs build the sandbox image and pick up ``ExecutionVerifier``.
    """
    from verifier import CodeVerifier, ExecutionVerifier  # type: ignore

    try:
        verifier = ExecutionVerifier()
        if verifier.image_available():
            return verifier
    except Exception:
        pass
    return CodeVerifier()


_CONFIGS: Dict[str, BenchmarkConfig] = {
    "gsm8k": BenchmarkConfig(
        name="gsm8k",
        problem_type="math",
        verifier_factory=_make_gsm8k_verifier,
        bin_for_problem=gsm8k_bin_for_problem,
        bin_names=("low", "mid", "high"),
    ),
    "math": BenchmarkConfig(
        name="math",
        problem_type="math",
        verifier_factory=_make_math_verifier,
        bin_for_problem=math_bin_for_problem,
        bin_names=MATH_LEVEL_BINS,
    ),
    "humaneval": BenchmarkConfig(
        name="humaneval",
        problem_type="code",
        verifier_factory=_make_humaneval_verifier,
        bin_for_problem=uniform_bin,
        bin_names=(UNIFORM_BIN,),
    ),
}


def benchmark_config(name: str) -> BenchmarkConfig:
    """Look up the benchmark wiring. Raises ``ValueError`` for unknown names."""
    key = name.lower()
    if key not in _CONFIGS:
        raise ValueError(
            f"Unknown benchmark {name!r}; expected one of {BENCHMARK_NAMES}"
        )
    return _CONFIGS[key]


def empty_bins_for(name: str) -> Dict[str, List[int]]:
    """Pre-seeded indices map for one benchmark's bin set."""
    cfg = benchmark_config(name)
    return {bucket: [] for bucket in cfg.bin_names}


def bins_for_problems(name: str, problems: Iterable) -> Dict[str, List[str]]:
    """Group problem ids by bucket using the benchmark's binner."""
    cfg = benchmark_config(name)
    out: Dict[str, List[str]] = {bucket: [] for bucket in cfg.bin_names}
    for problem in problems:
        bucket = cfg.bin_for_problem(problem)
        out.setdefault(bucket, []).append(problem.id)
    return out


__all__ = [
    "BENCHMARK_NAMES",
    "BenchmarkConfig",
    "benchmark_config",
    "empty_bins_for",
    "bins_for_problems",
]
