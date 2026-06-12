"""Bootstrap CI sanity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation.agreement.metrics import (
    agreement_rate,
    bootstrap_ci,
    cohens_kappa,
)


def test_ci_brackets_point_estimate():
    a = [True, False] * 50
    b = [True, False] * 50  # perfect agreement
    lo, hi = bootstrap_ci(cohens_kappa, a, b, n_resamples=200, seed=1)
    # the point estimate is 1.0; the CI should be at 1.0 because every
    # resample of identical lists is still identical
    assert lo == 1.0
    assert hi == 1.0


def test_ci_widens_for_noisy_data():
    import random
    rng = random.Random(0)
    n = 60
    a = [rng.random() > 0.4 for _ in range(n)]
    b = [a[i] if rng.random() > 0.25 else (not a[i]) for i in range(n)]
    lo, hi = bootstrap_ci(cohens_kappa, a, b, n_resamples=500, seed=7)
    assert lo <= cohens_kappa(a, b) <= hi
    assert hi - lo > 0.01  # nontrivial width


def test_invalid_args():
    import pytest

    with pytest.raises(ValueError):
        bootstrap_ci(agreement_rate)
    with pytest.raises(ValueError):
        bootstrap_ci(agreement_rate, [True], [True, False])
    with pytest.raises(ValueError):
        bootstrap_ci(agreement_rate, [True, False], [True, False], n_resamples=0)


def test_empty_returns_zero_zero():
    lo, hi = bootstrap_ci(cohens_kappa, [], [], n_resamples=10)
    assert (lo, hi) == (0.0, 0.0)
