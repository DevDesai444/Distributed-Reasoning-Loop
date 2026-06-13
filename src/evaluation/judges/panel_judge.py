"""
Panel judge — runs N judges (different models or different temperatures of the
same model) and takes a majority vote.

The panel members are constructed by the caller. We don't try to be clever
about composition here; that lives in the CLI / config layer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from .base import Judge, JudgeVerdict, Verdict


@dataclass
class PanelConfig:
    require_majority: bool = True  # if False, plurality wins
    tie_break: str = "uncertain"   # uncertain | reject | accept


class PanelJudge(Judge):
    """Aggregates several Judge instances by majority vote."""

    def __init__(self, members: List[Judge], config: Optional[PanelConfig] = None):
        if not members:
            raise ValueError("PanelJudge requires at least one member")
        self.members = members
        self.config = config or PanelConfig()
        self.judge_id = "panel::" + ",".join(m.judge_id for m in members)

    def judge(
        self,
        *,
        problem: str,
        response: str,
        reference_answer: Optional[str] = None,
        problem_type: str = "math",
    ) -> JudgeVerdict:
        per_member: List[JudgeVerdict] = []
        for member in self.members:
            per_member.append(
                member.judge(
                    problem=problem,
                    response=response,
                    reference_answer=reference_answer,
                    problem_type=problem_type,
                )
            )

        votes = Counter(v.verdict for v in per_member)
        # majority rule: needs > n/2
        n = len(per_member)
        top_verdict, top_count = votes.most_common(1)[0]

        if self.config.require_majority and top_count * 2 <= n:
            tie_map = {
                "uncertain": Verdict.UNCERTAIN,
                "reject": Verdict.REJECT,
                "accept": Verdict.ACCEPT,
            }
            final_verdict = tie_map.get(self.config.tie_break, Verdict.UNCERTAIN)
        else:
            final_verdict = top_verdict

        # confidence = mean confidence of members voting for the final verdict
        agreeing = [v for v in per_member if v.verdict == final_verdict]
        if agreeing:
            confidence = sum(v.confidence for v in agreeing) / len(agreeing)
        else:
            confidence = sum(v.confidence for v in per_member) / max(len(per_member), 1)

        # rationale = pick the most confident agreeing member, or note disagreement
        if agreeing:
            spokesperson = max(agreeing, key=lambda v: v.confidence)
            rationale = spokesperson.rationale
            reason_code = spokesperson.reason_code
        else:
            rationale = "panel split; no majority verdict"
            reason_code = "OTHER" if final_verdict == Verdict.REJECT else "NONE"

        return JudgeVerdict(
            verdict=final_verdict,
            confidence=confidence,
            rationale=rationale,
            reason_code=reason_code,
            raw_response="",
            judge_id=self.judge_id,
            prompt_version=per_member[0].prompt_version if per_member else "",
            metadata={
                "panel_size": n,
                "votes": {k.value: v for k, v in votes.items()},
                "per_member": [v.to_dict() for v in per_member],
            },
        )
