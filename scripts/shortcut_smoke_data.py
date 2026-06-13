#!/usr/bin/env python3
"""
Generate a deterministic synthetic shortcut + clean dataset.

This exists so the shortcut-PRM trainer can be exercised end-to-end without
depending on a real agreement run. It writes two JSONL files in the format
the dataset builder expects:

- ``samples/shortcut_prm_smoke/shortcuts.jsonl`` (label 1 candidates)
- ``samples/shortcut_prm_smoke/correct_samples.jsonl`` (label 0 candidates)

The smoke set uses 50 simple arithmetic problems. For each problem we emit:
- one "clean" trace with step-by-step reasoning and ``<<...>>`` blocks
- one "shortcut" trace following one of two templates:
    * INCOMPLETE_REASONING — final answer is correct but the middle is
      garbled
    * LUCKY_GUESS — the answer is asserted without derivation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


N_PROBLEMS = 50
DEFAULT_OUT_DIR = Path("./samples/shortcut_prm_smoke")


def _problem_for(idx: int) -> Tuple[str, int, int, int]:
    """
    Deterministic arithmetic problem generator.

    Returns ``(problem_text, a, b, answer)``. We alternate between additive
    and multiplicative templates so the smoke set has a little variation.
    """
    a = (idx * 3 + 7) % 47 + 1
    b = (idx * 5 + 13) % 31 + 1
    if idx % 2 == 0:
        problem = f"What is {a} plus {b}?"
        answer = a + b
    else:
        problem = f"A box has {a} rows and {b} items per row. Total items?"
        answer = a * b
    return problem, a, b, answer


def _clean_trace(a: int, b: int, answer: int, operator: str) -> str:
    if operator == "+":
        body = (
            f"Step 1: start with {a}.\n"
            f"Step 2: add {b}.\n"
            f"Step 3: <<{a}+{b}={answer}>>\n"
            f"#### {answer}"
        )
    else:
        body = (
            f"Step 1: there are {a} rows of {b}.\n"
            f"Step 2: total = {a} * {b}.\n"
            f"Step 3: <<{a}*{b}={answer}>>\n"
            f"#### {answer}"
        )
    return body


def _shortcut_trace(
    a: int, b: int, answer: int, operator: str, variant: str
) -> Tuple[str, str]:
    """Return ``(trace_text, reason_code)`` for a shortcut template."""
    if variant == "lucky_guess":
        text = f"The answer is {answer}.\n#### {answer}"
        return text, "LUCKY_GUESS"
    # incomplete_reasoning
    if operator == "+":
        garbled_mid = f"so something something {a} {b} blah."
    else:
        garbled_mid = f"so multiply-ish {a} blah {b} blah."
    text = (
        f"Step 1: numbers given.\n"
        f"Step 2: {garbled_mid}\n"
        f"Step 3: skipped.\n"
        f"#### {answer}"
    )
    return text, "INCOMPLETE_REASONING"


def build_smoke_records(
    n: int = N_PROBLEMS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(shortcut_records, clean_records)``."""
    shortcut_records: List[Dict[str, Any]] = []
    clean_records: List[Dict[str, Any]] = []
    for idx in range(n):
        problem, a, b, answer = _problem_for(idx)
        operator = "+" if idx % 2 == 0 else "*"
        clean_text = _clean_trace(a, b, answer, operator)
        variant = "lucky_guess" if idx % 2 == 0 else "incomplete_reasoning"
        shortcut_text, reason_code = _shortcut_trace(a, b, answer, operator, variant)

        problem_id = f"smoke_{idx:03d}"
        shortcut_records.append(
            {
                "problem_id": f"{problem_id}_shortcut",
                "pair_id": f"{problem_id}_shortcut",
                "problem": problem,
                "response_excerpt": shortcut_text,
                "reasoning": shortcut_text,
                "reason_code": reason_code,
                "judge_rationale": f"synthetic {reason_code.lower()} template",
                "judge_confidence": 0.85,
                "judge_id": "synthetic",
                "prompt_version": "smoke-v1",
            }
        )
        clean_records.append(
            {
                "problem_id": f"{problem_id}_clean",
                "pair_id": f"{problem_id}_clean",
                "problem": problem,
                "reasoning": clean_text,
                "final_answer": str(answer),
            }
        )
    return shortcut_records, clean_records


def write_smoke_dataset(out_dir: Path, n: int = N_PROBLEMS) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shortcut_records, clean_records = build_smoke_records(n=n)
    shortcuts_path = out_dir / "shortcuts.jsonl"
    clean_path = out_dir / "correct_samples.jsonl"
    with open(shortcuts_path, "w") as handle:
        for row in shortcut_records:
            handle.write(json.dumps(row) + "\n")
    with open(clean_path, "w") as handle:
        for row in clean_records:
            handle.write(json.dumps(row) + "\n")
    return {"shortcuts": shortcuts_path, "clean": clean_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic shortcut + clean smoke dataset.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n", type=int, default=N_PROBLEMS)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    paths = write_smoke_dataset(out_dir, n=args.n)
    logger.info("Wrote smoke dataset to %s", out_dir)
    print(f"shortcuts -> {paths['shortcuts']}")
    print(f"clean     -> {paths['clean']}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
