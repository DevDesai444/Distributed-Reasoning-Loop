"""
Pairwise judge — picks the better of two candidate solutions.

Position bias is the well-known failure mode here. We run both orderings and
report stability; if `swap_to_resolve` is on, we only emit a winner when the
two orderings agree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .prompts import get_rubric
from .single_model_judge import SingleModelJudge


@dataclass
class PairwiseVerdict:
    winner: str  # "A", "B", or "TIE"
    confidence: float
    rationale: str
    position_swap_stable: bool
    judge_id: str
    prompt_version: str


def _parse_pairwise(text: str) -> dict:
    winner_match = re.search(r"WINNER\s*[:\-]?\s*(A|B|TIE)", text or "", flags=re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE\s*[:\-]?\s*([0-9.]+)", text or "", flags=re.IGNORECASE)
    rat_match = re.search(r"RATIONALE\s*[:\-]?\s*(.+?)(?:\n[A-Z_]+\s*[:\-]|\Z)", text or "", flags=re.IGNORECASE | re.DOTALL)
    winner = (winner_match.group(1).upper() if winner_match else "TIE")
    try:
        conf = float(conf_match.group(1)) if conf_match else 0.0
    except ValueError:
        conf = 0.0
    rationale = rat_match.group(1).strip() if rat_match else ""
    return {"winner": winner, "confidence": max(0.0, min(1.0, conf)), "rationale": rationale}


class PairwiseJudge:
    """Wraps a SingleModelJudge backend; uses the pairwise rubric."""

    def __init__(self, base_judge: SingleModelJudge):
        self.base_judge = base_judge
        self.judge_id = f"pairwise::{base_judge.config.model_name}"

    def _ask(self, problem: str, a: str, b: str) -> dict:
        version_tag, template = get_rubric("pairwise")
        prompt = template.format(problem=problem, response_a=a, response_b=b)
        # reuse the single-judge generation backend
        raw = self.base_judge._generate(prompt)  # type: ignore[attr-defined]
        return {"version_tag": version_tag, "raw": raw, **_parse_pairwise(raw)}

    def judge_pair(
        self,
        *,
        problem: str,
        response_a: str,
        response_b: str,
        check_position_swap: bool = True,
    ) -> PairwiseVerdict:
        forward = self._ask(problem, response_a, response_b)
        if not check_position_swap:
            return PairwiseVerdict(
                winner=forward["winner"],
                confidence=forward["confidence"],
                rationale=forward["rationale"],
                position_swap_stable=True,
                judge_id=self.judge_id,
                prompt_version=forward["version_tag"],
            )

        reverse = self._ask(problem, response_b, response_a)
        # In reverse, A and B are swapped: a "B" winner in reverse maps to A in
        # the original order. TIE stays TIE.
        reverse_to_forward = {"A": "B", "B": "A", "TIE": "TIE"}
        reverse_in_forward = reverse_to_forward[reverse["winner"]]
        stable = forward["winner"] == reverse_in_forward

        # If unstable, downgrade to TIE with halved confidence so downstream
        # metrics know the call was shaky.
        if stable:
            winner = forward["winner"]
            confidence = (forward["confidence"] + reverse["confidence"]) / 2
        else:
            winner = "TIE"
            confidence = max(forward["confidence"], reverse["confidence"]) / 2

        rationale = forward["rationale"]
        if not stable:
            rationale = f"position-swap unstable; forward={forward['winner']} reverse={reverse_in_forward}. " + rationale

        return PairwiseVerdict(
            winner=winner,
            confidence=confidence,
            rationale=rationale,
            position_swap_stable=stable,
            judge_id=self.judge_id,
            prompt_version=forward["version_tag"],
        )
