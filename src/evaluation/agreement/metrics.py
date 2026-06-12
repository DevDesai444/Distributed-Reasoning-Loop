"""
Agreement metrics for comparing two binary evaluators over the same trace set.

All functions take parallel sequences of booleans (or Nones for "unknown" /
"uncertain"). None values are treated as missing — they're filtered out before
metric computation. We log the drop count in confusion_matrix so the caller
can see how many traces fell out.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Verdict = Optional[bool]


def _filter_pairs(
    a: Sequence[Verdict],
    b: Sequence[Verdict],
) -> Tuple[List[bool], List[bool], int]:
    if len(a) != len(b):
        raise ValueError("verdict sequences must be the same length")
    fa: List[bool] = []
    fb: List[bool] = []
    dropped = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            dropped += 1
            continue
        fa.append(bool(x))
        fb.append(bool(y))
    return fa, fb, dropped


def agreement_rate(
    verdicts_a: Sequence[Verdict],
    verdicts_b: Sequence[Verdict],
) -> float:
    """Raw fraction of pairs that match."""
    fa, fb, _ = _filter_pairs(verdicts_a, verdicts_b)
    if not fa:
        return 0.0
    matches = sum(1 for x, y in zip(fa, fb) if x == y)
    return matches / len(fa)


def cohens_kappa(
    verdicts_a: Sequence[Verdict],
    verdicts_b: Sequence[Verdict],
) -> float:
    """
    Cohen's kappa for two binary raters.

    Returns 1.0 when both lists agree perfectly and there's any class
    variation. Returns 0.0 when both raters give a single constant label —
    technically kappa is undefined (chance agreement == observed agreement),
    but reporting 0 matches what every downstream chart expects.
    """
    fa, fb, _ = _filter_pairs(verdicts_a, verdicts_b)
    n = len(fa)
    if n == 0:
        return 0.0

    # observed agreement
    po = sum(1 for x, y in zip(fa, fb) if x == y) / n

    # marginals
    pa_true = sum(fa) / n
    pb_true = sum(fb) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)

    if abs(1.0 - pe) < 1e-12:
        # constant labels — return 0 (some libs return NaN; we prefer a number)
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1 - pe)


def confusion_matrix(
    verdicts_a: Sequence[Verdict],
    verdicts_b: Sequence[Verdict],
) -> Dict[str, int]:
    """
    Confusion matrix keyed by (a_label, b_label) where labels are
    accept/reject. Reports dropped uncertain pairs separately.

    Convention: rows are evaluator A (treated as reference for TP/FP labels),
    columns are evaluator B.
    - TP = both accept
    - TN = both reject
    - FP = A reject, B accept (B says yes, ref says no)
    - FN = A accept, B reject (B says no, ref says yes)
    """
    fa, fb, dropped = _filter_pairs(verdicts_a, verdicts_b)
    tp = tn = fp = fn = 0
    for x, y in zip(fa, fb):
        if x and y:
            tp += 1
        elif (not x) and (not y):
            tn += 1
        elif (not x) and y:
            fp += 1
        else:
            fn += 1
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": len(fa),
        "dropped_uncertain": dropped,
    }


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *args: Sequence[Verdict],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = 0,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for a metric that takes parallel sequences.

    Returns (lower, upper). All input sequences are resampled with the same
    index draws so paired structure is preserved.
    """
    if not args:
        raise ValueError("bootstrap_ci needs at least one sequence")
    n = len(args[0])
    if any(len(seq) != n for seq in args):
        raise ValueError("all input sequences must be the same length")
    if n == 0:
        return (0.0, 0.0)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")

    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        resampled = tuple([seq[i] for i in idx] for seq in args)
        try:
            samples.append(float(metric_fn(*resampled)))
        except Exception:
            # if a resample degenerates (e.g. all-uncertain), skip it
            continue

    if not samples:
        return (0.0, 0.0)

    samples.sort()
    alpha = (1 - ci) / 2
    lo_idx = max(0, int(math.floor(alpha * len(samples))))
    hi_idx = min(len(samples) - 1, int(math.ceil((1 - alpha) * len(samples))) - 1)
    return (samples[lo_idx], samples[hi_idx])
