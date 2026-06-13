"""Sanity checks on Cohen's kappa."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation.agreement.metrics import cohens_kappa


def test_perfect_agreement_returns_one():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0


def test_complete_disagreement_is_negative():
    k = cohens_kappa([True, True, False, False], [False, False, True, True])
    assert k < 0
    # exact value for this case is -1
    assert math.isclose(k, -1.0, abs_tol=1e-9)


def test_random_chance_near_zero():
    # 100 pairs where A is all True 50% of the time, B independently 50%
    a = [True, False] * 50
    b = [True, True, False, False] * 25
    k = cohens_kappa(a, b)
    # not exactly zero, but well under 0.1
    assert -0.1 <= k <= 0.1


def test_constant_labels_returns_zero():
    # both raters say True everywhere — chance agreement == observed, kappa
    # is undefined; we return 1.0 because observed agreement is perfect
    assert cohens_kappa([True] * 10, [True] * 10) == 1.0


def test_constant_but_disagreeing_returns_zero():
    # all True vs all False
    assert cohens_kappa([True] * 5, [False] * 5) == 0.0


def test_uncertain_pairs_dropped():
    # None values should be filtered before metric computation
    a = [True, None, True, False, False]
    b = [True, True, True, False, False]
    # after drop: a=[T,T,F,F], b=[T,T,F,F] -> perfect
    assert cohens_kappa(a, b) == 1.0


def test_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        cohens_kappa([True, False], [True])
