"""Tests for the per-benchmark agreement wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluation.agreement.benchmark_config import (  # noqa: E402
    BENCHMARK_NAMES,
    benchmark_config,
    bins_for_problems,
    empty_bins_for,
)


def test_known_benchmarks():
    assert set(BENCHMARK_NAMES) == {"gsm8k", "math", "humaneval"}
    for name in BENCHMARK_NAMES:
        cfg = benchmark_config(name)
        assert cfg.name == name
        assert callable(cfg.verifier_factory)
        assert callable(cfg.bin_for_problem)
        assert cfg.bin_names  # non-empty


def test_problem_type_routing():
    assert benchmark_config("gsm8k").problem_type == "math"
    assert benchmark_config("math").problem_type == "math"
    assert benchmark_config("humaneval").problem_type == "code"


def test_gsm8k_verifier_is_numeric():
    from verifier import GSM8KVerifier, MathVerifier

    cfg = benchmark_config("gsm8k")
    verifier = cfg.verifier_factory()
    assert isinstance(verifier, GSM8KVerifier)
    # GSM8KVerifier inherits MathVerifier with numeric specialization.
    assert isinstance(verifier, MathVerifier)


def test_math_verifier_supports_symbolic():
    from verifier import MathVerifier

    cfg = benchmark_config("math")
    verifier = cfg.verifier_factory()
    assert isinstance(verifier, MathVerifier)
    # The MathVerifier class natively offers compare_symbolic — that's
    # the contract MATH dispatch relies on.
    assert hasattr(verifier, "compare_symbolic")


def test_humaneval_verifier_falls_back_without_docker():
    from verifier import CodeVerifier, ExecutionVerifier

    cfg = benchmark_config("humaneval")
    verifier = cfg.verifier_factory()
    # On a CPU host without the sandbox image, we must fall back to a
    # CodeVerifier-compatible interface so the smoke can still run.
    assert isinstance(verifier, (CodeVerifier, ExecutionVerifier))


def test_gsm8k_binner_uses_step_count():
    cfg = benchmark_config("gsm8k")
    p_low = SimpleNamespace(id="a", metadata={"full_solution": "x = <<1+1=2>>. #### 2"})
    p_high = SimpleNamespace(id="b", metadata={"full_solution": "<<1+1=2>>" * 9 + " #### 18"})
    assert cfg.bin_for_problem(p_low) == "low"
    assert cfg.bin_for_problem(p_high) == "high"


def test_math_binner_uses_level():
    cfg = benchmark_config("math")
    p = SimpleNamespace(id="a", metadata={"level": "Level 4"})
    assert cfg.bin_for_problem(p) == "level_4"


def test_humaneval_binner_is_uniform():
    cfg = benchmark_config("humaneval")
    p = SimpleNamespace(id="he/0", metadata={})
    assert cfg.bin_for_problem(p) == "all"


def test_empty_bins_for_seeds_all_buckets():
    bins = empty_bins_for("math")
    assert set(bins.keys()) == set(benchmark_config("math").bin_names)
    assert all(v == [] for v in bins.values())


def test_bins_for_problems_groups_ids():
    problems = [
        SimpleNamespace(id="a", metadata={"level": "Level 1"}),
        SimpleNamespace(id="b", metadata={"level": "Level 5"}),
        SimpleNamespace(id="c", metadata={"level": "Level 1"}),
    ]
    grouped = bins_for_problems("math", problems)
    assert grouped["level_1"] == ["a", "c"]
    assert grouped["level_5"] == ["b"]


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError):
        benchmark_config("not-a-benchmark")
