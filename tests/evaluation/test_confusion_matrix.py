"""Confusion-matrix bookkeeping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation.agreement.metrics import confusion_matrix


def test_all_buckets_counted():
    # A=[T, T, F, F], B=[T, F, T, F]
    # tp=1, fn=1, fp=1, tn=1
    m = confusion_matrix([True, True, False, False], [True, False, True, False])
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert m["n"] == 4
    assert m["dropped_uncertain"] == 0


def test_dropped_uncertain_counted():
    m = confusion_matrix([True, None, True], [True, True, None])
    assert m["dropped_uncertain"] == 2
    assert m["n"] == 1
    assert m["tp"] == 1


def test_all_accept():
    m = confusion_matrix([True, True], [True, True])
    assert m["tp"] == 2 and m["tn"] == 0 and m["fp"] == 0 and m["fn"] == 0


def test_empty_input():
    m = confusion_matrix([], [])
    assert all(m[k] == 0 for k in ("tp", "tn", "fp", "fn", "n", "dropped_uncertain"))
