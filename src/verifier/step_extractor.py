"""
Utilities for extracting reasoning steps from free-form traces.
"""

from __future__ import annotations


def extract_steps(reasoning_text: str) -> list[str]:
    """
    Split reasoning into newline-separated steps and discard short fragments.
    """
    steps: list[str] = []
    for raw_line in reasoning_text.splitlines():
        cleaned = " ".join(raw_line.strip().split())
        if len(cleaned) >= 15:
            steps.append(cleaned)
    return steps
