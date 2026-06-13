"""
Bias-control utilities for judge runs.

These are deliberately small, pure functions so the metrics layer can call
them without dragging a model along.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .base import Verdict


@dataclass
class PositionSwapResult:
    n_pairs: int
    n_stable: int
    swap_rate: float  # fraction unstable

    @property
    def stability(self) -> float:
        return self.n_stable / self.n_pairs if self.n_pairs else 0.0


def position_swap_stability(
    forward_winners: Sequence[str],
    reverse_winners: Sequence[str],
) -> PositionSwapResult:
    """
    Given pairwise verdicts collected with A/B in forward order and with the
    candidates swapped, count how often the judge picked the same underlying
    response.

    Verdicts are "A", "B", or "TIE" in each ordering.
    """
    if len(forward_winners) != len(reverse_winners):
        raise ValueError("forward/reverse winner lists must be the same length")

    swap_map = {"A": "B", "B": "A", "TIE": "TIE"}
    stable = 0
    for fwd, rev in zip(forward_winners, reverse_winners):
        if swap_map[rev] == fwd:
            stable += 1
    n = len(forward_winners)
    return PositionSwapResult(
        n_pairs=n,
        n_stable=stable,
        swap_rate=(n - stable) / n if n else 0.0,
    )


def truncate_to_min_length(*responses: str) -> Tuple[str, ...]:
    """
    Truncate all inputs to the shortest length. Used to neutralize length as a
    cue when probing whether the judge favors longer answers.
    """
    if not responses:
        return tuple()
    min_len = min(len(r) for r in responses)
    return tuple(r[:min_len] for r in responses)


@dataclass
class SelfPreferenceResult:
    n_problems: int
    self_accept_rate: float
    other_accept_rate: float
    delta: float


def self_preference_delta(
    self_verdicts: Sequence[Verdict],
    other_verdicts: Sequence[Verdict],
) -> SelfPreferenceResult:
    """
    Compare accept-rate of a judge on its own outputs vs on a peer model's
    outputs over a fixed problem set. A large positive delta is evidence of
    self-preference bias.
    """
    if len(self_verdicts) != len(other_verdicts):
        raise ValueError("self/other verdict lists must be the same length")

    def _rate(verdicts: Sequence[Verdict]) -> float:
        if not verdicts:
            return 0.0
        return sum(1 for v in verdicts if v == Verdict.ACCEPT) / len(verdicts)

    s_rate = _rate(self_verdicts)
    o_rate = _rate(other_verdicts)
    return SelfPreferenceResult(
        n_problems=len(self_verdicts),
        self_accept_rate=s_rate,
        other_accept_rate=o_rate,
        delta=s_rate - o_rate,
    )


def length_bias_probe(
    judge_fn: Callable[[str, str], Verdict],
    long_wrong: str,
    short_right: str,
) -> dict:
    """
    Synthetic probe: ask whether the judge marks a long-wrong answer ACCEPT
    while rejecting a short-right answer. Returns a small dict the test layer
    can inspect.

    `judge_fn` should be `(problem, response) -> Verdict`.
    """
    long_verdict = judge_fn("probe", long_wrong)
    short_verdict = judge_fn("probe", short_right)
    return {
        "long_wrong_verdict": long_verdict.value,
        "short_right_verdict": short_verdict.value,
        "length_bias_detected": (
            long_verdict == Verdict.ACCEPT and short_verdict == Verdict.REJECT
        ),
    }
