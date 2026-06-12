"""
Composite quality score for synthetic reasoning traces.

The score is a simple average of three components, each in [0, 1]:

1. Verifier component — 1.0 if the trace's final answer matches the gold
   answer (ACCEPT), 0.0 otherwise. This is a placeholder until we wire a
   process reward model; right now the verifier is binary so the component
   collapses to the verdict.
2. Step coherence — for every inline `<<a op b = c>>` arithmetic block we
   check the equation. Returns the fraction of blocks that check out. If
   there are no such blocks we return 1.0 (no evidence of inconsistency).
3. Length penalty — penalize degenerate, run-on traces. Full credit when
   length is at or below 2x median; smoothly drops to 0.5 at 4x median;
   floors at 0.0 well past that.

The score is used downstream as a filtering threshold; we keep it cheap to
compute so it can run inline during pair construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Matches GSM8K-style inline calc tags, e.g. <<48/2=24>> or <<48 + 24 = 72>>.
# We accept +, -, *, /, x (as multiplication) with optional whitespace.
_STEP_RE = re.compile(
    r"<<\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*([+\-*/x×])\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*=\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*>>"
)


@dataclass
class QualityComponents:
    verifier: float
    step_coherence: float
    length_penalty: float

    @property
    def score(self) -> float:
        return (self.verifier + self.step_coherence + self.length_penalty) / 3.0


def _safe_eval(a: float, op: str, b: float) -> Optional[float]:
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op in ("*", "x", "×"):  # × is "×"
            return a * b
        if op == "/":
            if b == 0:
                return None
            return a / b
    except Exception:
        return None
    return None


def step_coherence(trace_text: str) -> float:
    """Fraction of inline `<<...>>` arithmetic blocks that check out.

    Returns 1.0 when there are no such blocks: absence of evidence is not
    evidence of incoherence, and we don't want to penalize traces that
    happen not to use the calc-tag convention.
    """
    matches = _STEP_RE.findall(trace_text or "")
    if not matches:
        return 1.0
    ok = 0
    tol = 1e-6
    for a_str, op, b_str, c_str in matches:
        a = float(a_str)
        b = float(b_str)
        c = float(c_str)
        expected = _safe_eval(a, op, b)
        if expected is None:
            continue
        if abs(expected - c) <= max(tol, tol * max(abs(expected), 1.0)):
            ok += 1
    return ok / len(matches)


def length_penalty(length: int, median_length: int) -> float:
    """Smooth penalty: 1.0 at <=2x median, 0.5 at 4x median, 0.0 by 6x."""
    if median_length <= 0 or length <= 0:
        return 1.0
    ratio = length / float(median_length)
    if ratio <= 2.0:
        return 1.0
    if ratio >= 6.0:
        return 0.0
    if ratio <= 4.0:
        # Linear 1.0 -> 0.5 between 2x and 4x
        return 1.0 - 0.5 * (ratio - 2.0) / 2.0
    # Linear 0.5 -> 0.0 between 4x and 6x
    return 0.5 - 0.5 * (ratio - 4.0) / 2.0


def score_components(
    trace_text: str,
    is_correct: bool,
    median_length: int,
) -> QualityComponents:
    """Compute the three components without averaging."""
    verifier = 1.0 if is_correct else 0.0
    coherence = step_coherence(trace_text)
    length = length_penalty(len(trace_text or ""), median_length)
    return QualityComponents(
        verifier=verifier,
        step_coherence=coherence,
        length_penalty=length,
    )


def score_trace(
    trace_text: str,
    is_correct: bool,
    median_length: int,
) -> float:
    """Composite score in [0, 1]. Mean of three [0, 1] components."""
    return score_components(trace_text, is_correct, median_length).score


__all__ = [
    "QualityComponents",
    "score_components",
    "score_trace",
    "step_coherence",
    "length_penalty",
]
