"""
Tests for reasoning-step extraction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verifier.step_extractor import extract_steps


def test_extract_steps_filters_short_lines():
    reasoning = """
    short
    Step 1: Multiply both sides by 2 to isolate the variable.
    x = 2
    Therefore the final answer is 8 after substitution.
    """

    steps = extract_steps(reasoning)

    assert steps == [
        "Step 1: Multiply both sides by 2 to isolate the variable.",
        "Therefore the final answer is 8 after substitution.",
    ]
