"""Tests for the MATH difficulty binner."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluation.agreement import math_difficulty_binner as binner  # noqa: E402
from evaluation.agreement.math_difficulty_binner import (  # noqa: E402
    ALL_BINS,
    LEVELS,
    MathDifficultyBinner,
    UNKNOWN_BIN,
    bin_for_level,
    bin_for_problem,
    bin_problems,
    parse_level,
)


def _problem(level=None, pid="p"):
    metadata = {} if level is None else {"level": level}
    return SimpleNamespace(id=pid, metadata=metadata)


@pytest.mark.parametrize("raw,expected", [
    ("Level 1", 1),
    ("Level 5", 5),
    ("level 3", 3),
    ("3", 3),
    (4, 4),
    ("Level 7", None),
    (0, None),
    (-1, None),
    (None, None),
    ("", None),
    (True, None),
])
def test_parse_level_variants(raw, expected):
    assert parse_level(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Level 1", "level_1"),
    ("Level 5", "level_5"),
    (3, "level_3"),
    ("garbage", UNKNOWN_BIN),
    (None, UNKNOWN_BIN),
])
def test_bin_for_level(raw, expected):
    assert bin_for_level(raw) == expected


def test_bin_for_problem_missing_field_is_unknown():
    p = _problem()
    assert bin_for_problem(p) == UNKNOWN_BIN


def test_bin_for_problem_metadata_none_is_unknown():
    p = SimpleNamespace(id="p", metadata=None)
    assert bin_for_problem(p) == UNKNOWN_BIN


def test_bin_problems_groups_by_level():
    problems = [
        _problem("Level 1", pid="a"),
        _problem("Level 2", pid="b"),
        _problem("Level 5", pid="c"),
        _problem("Level 5", pid="d"),
        _problem(None, pid="e"),
    ]
    grouped = bin_problems(problems)
    assert grouped["level_1"] == ["a"]
    assert grouped["level_2"] == ["b"]
    assert grouped["level_5"] == ["c", "d"]
    assert grouped[UNKNOWN_BIN] == ["e"]
    # Untouched buckets remain present so downstream callers can iterate.
    assert grouped["level_3"] == []


def test_all_bins_constant_matches_levels():
    assert ALL_BINS == LEVELS + (UNKNOWN_BIN,)


def test_binner_object_matches_module_functions():
    b = MathDifficultyBinner()
    assert b.bins == ALL_BINS
    p = _problem("Level 4", pid="z")
    assert b.bin_for_problem(p) == bin_for_problem(p) == "level_4"
    grouped = b.bin_problems([p])
    assert grouped["level_4"] == ["z"]


def test_module_exposes_uniform_binner():
    # Sibling helper from difficulty_binner.py; smoke check that the
    # public surface used by the agreement runner is in place.
    from evaluation.agreement import UNIFORM_BIN, UniformBinner, uniform_bin

    assert UNIFORM_BIN == "all"
    assert uniform_bin(_problem()) == UNIFORM_BIN
    ub = UniformBinner()
    assert ub.bins == (UNIFORM_BIN,)
    assert ub.bin_for_problem(_problem()) == UNIFORM_BIN
