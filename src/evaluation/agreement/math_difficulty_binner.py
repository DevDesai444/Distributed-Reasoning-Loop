"""
Difficulty bins for the MATH benchmark.

MATH problems carry a `level` field in the loader metadata; values are
strings like ``"Level 1"`` through ``"Level 5"`` (1 = easiest).
The bin label is ``level_1`` ... ``level_5``. When the field is missing
or unparsable we drop the trace into ``level_unknown`` so the run
keeps going on partial datasets.

API mirrors `difficulty_binner.bin_for_problem` so the agreement runner
can swap the function in without further branching.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


LEVELS = ("level_1", "level_2", "level_3", "level_4", "level_5")
UNKNOWN_BIN = "level_unknown"
ALL_BINS = LEVELS + (UNKNOWN_BIN,)

_LEVEL_PATTERN = re.compile(r"(\d+)")


def parse_level(raw: Optional[object]) -> Optional[int]:
    """Coerce a MATH `level` field into an integer 1..5, or None."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if 1 <= raw <= 5 else None
    text = str(raw)
    match = _LEVEL_PATTERN.search(text)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    if 1 <= value <= 5:
        return value
    return None


def bin_for_level(raw: Optional[object]) -> str:
    """Map a raw `level` field to a bin name; falls back to `level_unknown`."""
    parsed = parse_level(raw)
    if parsed is None:
        return UNKNOWN_BIN
    return f"level_{parsed}"


def bin_for_problem(problem) -> str:
    """Return the MATH difficulty bin for one Problem-like object."""
    meta = getattr(problem, "metadata", None) or {}
    return bin_for_level(meta.get("level"))


def bin_problems(problems: Iterable) -> Dict[str, List[str]]:
    """Group problem ids by MATH level bin."""
    buckets: Dict[str, List[str]] = {name: [] for name in ALL_BINS}
    for problem in problems:
        bucket = bin_for_problem(problem)
        buckets[bucket].append(problem.id)
    return buckets


class MathDifficultyBinner:
    """Drop-in object form of the binner; matches the GSM8K binner shape."""

    bins = ALL_BINS

    def bin_for_problem(self, problem) -> str:  # noqa: D401
        return bin_for_problem(problem)

    def bin_problems(self, problems: Iterable) -> Dict[str, List[str]]:
        return bin_problems(problems)
