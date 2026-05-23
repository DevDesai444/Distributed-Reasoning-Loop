"""
Evaluation module for reasoning models.
Includes Test-Time Compute, Best-of-N sampling, and benchmarking.
"""

CORE_EVALUATION_COMPONENTS = [
    "GSM8KEvaluator",
    "HumanEvalEvaluator",
    "MATHEvaluator",
    "BenchmarkResult",
    "run_all_benchmarks",
    "TestTimeCompute",
    "BestOfNSampler",
]

OPTIONAL_EXPERIMENTAL_COMPONENTS = {
    "search_strategies": [
        "BeamSearchReasoner",
        "MCTSReasoner",
    ],
}

from .test_time_compute import (
    TestTimeCompute,
    BestOfNSampler,
    BeamSearchReasoner,
    MCTSReasoner,
)

from .benchmarks import (
    GSM8KEvaluator,
    HumanEvalEvaluator,
    MATHEvaluator,
    BenchmarkResult,
    run_all_benchmarks,
)

__all__ = [
    # Test-Time Compute
    "TestTimeCompute",
    "BestOfNSampler",
    "BeamSearchReasoner",
    "MCTSReasoner",
    # Benchmarks
    "GSM8KEvaluator",
    "HumanEvalEvaluator",
    "MATHEvaluator",
    "BenchmarkResult",
    "run_all_benchmarks",
    "CORE_EVALUATION_COMPONENTS",
    "OPTIONAL_EXPERIMENTAL_COMPONENTS",
]
