#!/usr/bin/env python3
"""
Pipeline smoke test for the robustness eval.

Does not benchmark anything. Builds a handful of GSM8K-style problems
inline, crafts deterministic stub "model outputs" that always answer
correctly when the gold appears in the prompt context, and runs the full
runner. Output goes to `samples/robustness_smoke/` and is committed as a
pipeline artifact — not a benchmark result.

Real runs go through `python main.py eval robust ...`.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.robustness import default_perturbations, evaluate_robustness  # noqa: E402
from verifier import GSM8KVerifier, VerificationStatus  # type: ignore  # noqa: E402


@dataclass
class _Problem:
    id: str
    problem: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


SYNTH_PROBLEMS: List[_Problem] = [
    _Problem(
        id="smoke_1",
        problem=(
            "Sarah buys 4 packs of stickers. Each pack costs 3 dollars. "
            "How much does she spend?"
        ),
        answer="12",
        metadata={"full_solution": "She spends 4 * 3 = <<4*3=12>>12 dollars.\n#### 12"},
    ),
    _Problem(
        id="smoke_2",
        problem=(
            "A train travels 60 miles per hour. The trip lasts 2 hours. "
            "How many miles total?"
        ),
        answer="120",
        metadata={"full_solution": "<<60*2=120>>120\n#### 120"},
    ),
    _Problem(
        id="smoke_3",
        problem=(
            "Tom has 4 marbles. Jerry has 3 marbles. "
            "How many marbles do they have altogether?"
        ),
        answer="7",
        metadata={"full_solution": "<<4+3=7>>7\n#### 7"},
    ),
    _Problem(
        id="smoke_4",
        problem=(
            "A bottle holds 2 kg of sugar. The shop sells 5 bottles. "
            "How many kg total?"
        ),
        answer="10",
        metadata={"full_solution": "<<2*5=10>>10\n#### 10"},
    ),
    _Problem(
        id="smoke_5",
        problem=(
            "She buys 5 books and then reads them all. "
            "Each book has 100 pages. How many pages total?"
        ),
        answer="500",
        metadata={"full_solution": "<<5*100=500>>500\n#### 500"},
    ),
    _Problem(
        id="smoke_6",
        problem=(
            "A child has 6 candies. Their friend gives 4 more candies. "
            "How many candies do they have now?"
        ),
        answer="10",
        metadata={"full_solution": "<<6+4=10>>10\n#### 10"},
    ),
]


def _make_stub_generate(problems: List["_Problem"]):
    """
    Stub generator that "knows" the gold answer for each problem and finds
    the right answer by substring-matching the prompt against the original
    problem text. For label-rewriting perturbations (number_swap,
    unit_change) the original problem won't be a substring of the perturbed
    one, so the stub will fail — that's intentional, it simulates a model
    that doesn't generalize across rewrites.

    For label-preserving perturbations (irrelevant_context,
    distractor_sentence, paraphrase, reordering) the original text either
    appears verbatim somewhere in the perturbed prompt, or — for
    paraphrase/reordering — it doesn't, so those also "fail" the stub.
    Net effect: retention on irrelevant_context and distractor_sentence is
    high, retention on number_swap / unit_change / paraphrase / reordering
    is lower — a more honest smoke profile than uniform 1.0.
    """
    # Map first sentence of each problem -> gold answer
    first_sentence_to_answer: Dict[str, str] = {}
    for prob in problems:
        first = _split_first_sentence(prob.problem)
        first_sentence_to_answer[first] = prob.answer

    def generate(problem_text: str, _n: int) -> List[str]:
        # Find which original problem this came from.
        for sent, gold in first_sentence_to_answer.items():
            if sent in problem_text:
                return [f"The answer is {gold}\n#### {gold}"]
        # No match — emit nothing recognisable
        return ["I don't know.\n#### 0"]

    return generate


def _split_first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0]


def _verify(output: str, gold: str) -> Optional[bool]:
    verifier = GSM8KVerifier()
    try:
        result = verifier.verify_reasoning_path(output, gold)
        if result.status == VerificationStatus.CORRECT:
            return True
        if result.status == VerificationStatus.INCORRECT:
            return False
        return None
    except Exception:
        return None


def main() -> int:
    out_dir = Path(__file__).parent.parent / "samples" / "robustness_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    perturbations = default_perturbations()
    generate_fn = _make_stub_generate(SYNTH_PROBLEMS)

    report = evaluate_robustness(
        problems=SYNTH_PROBLEMS,
        n_samples=1,
        perturbations=perturbations,
        generate_fn=generate_fn,
        verify_fn=_verify,
        seed=0,
        n_bootstrap=200,
        model_label="smoke-stub",
        benchmark_label="gsm8k-synthetic",
        run_id="robustness_smoke",
        extra_metadata={"note": "synthetic stub; not a benchmark"},
    )

    (out_dir / "robustness_results.json").write_text(report.render_json())
    (out_dir / "report.md").write_text(report.render_markdown())

    manifest = {
        "run_id": "robustness_smoke",
        "kind": "robustness_smoke",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "smoke-stub",
        "benchmark": "gsm8k-synthetic",
        "n_problems": len(SYNTH_PROBLEMS),
        "n_samples_per_problem": 1,
        "seed": 0,
        "overall_robustness": report.overall_robustness,
        "note": "Pipeline smoke artifact, not a benchmark result.",
        "artifacts": ["robustness_results.json", "report.md"],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"smoke run written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
