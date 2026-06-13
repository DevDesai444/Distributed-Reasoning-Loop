"""
Difficulty bins for GSM8K based on the number of reasoning steps in the
reference solution.

GSM8K's gold solutions use `<<expression=result>>` markers per step. The
count of `<<` occurrences is a decent proxy for problem depth.

Bin thresholds:
- low:  <= 3 steps
- mid:  4 - 6 steps
- high: >= 7 steps
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


LOW_MAX = 3
MID_MAX = 6


def step_count_from_solution(solution: Optional[str]) -> int:
    """Count `<<` markers in a GSM8K-style solution. Falls back to 0."""
    if not solution:
        return 0
    return solution.count("<<")


def bin_for_step_count(steps: int) -> str:
    if steps <= LOW_MAX:
        return "low"
    if steps <= MID_MAX:
        return "mid"
    return "high"


def bin_problems(problems: Iterable) -> Dict[str, List[str]]:
    """
    Group problem ids by difficulty bin.

    Expects each problem to have `.id` and a `.metadata` dict carrying the
    full GSM8K solution string under `full_solution`. Anything missing falls
    into the "low" bucket — better than crashing on a partial dataset.
    """
    buckets: Dict[str, List[str]] = {"low": [], "mid": [], "high": []}
    for problem in problems:
        solution = None
        meta = getattr(problem, "metadata", None) or {}
        solution = meta.get("full_solution") or meta.get("solution")
        steps = step_count_from_solution(solution)
        bucket = bin_for_step_count(steps)
        buckets[bucket].append(problem.id)
    return buckets


def bin_for_problem(problem) -> str:
    meta = getattr(problem, "metadata", None) or {}
    solution = meta.get("full_solution") or meta.get("solution")
    return bin_for_step_count(step_count_from_solution(solution))
