"""
Shortcut detector: finds traces the verifier marks correct but the judge
flags as bad reasoning. These are the "right answer for the wrong reason"
cases — useful signal for whether the model is learning shortcuts.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .judges import REASON_CODES, JudgeVerdict, Verdict


@dataclass
class ShortcutRecord:
    problem_id: str
    problem: str
    response_excerpt: str
    judge_rationale: str
    reason_code: str
    judge_confidence: float
    judge_id: str
    prompt_version: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShortcutSummary:
    n_verifier_accepted: int
    n_shortcuts: int
    shortcut_rate: float
    by_reason: Dict[str, int]
    by_difficulty: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render_markdown(self, *, title: str = "Reasoning Shortcuts") -> str:
        lines = [f"# {title}", ""]
        lines.append(f"- Verifier-accepted traces: **{self.n_verifier_accepted}**")
        lines.append(f"- Shortcuts (judge rejected): **{self.n_shortcuts}**")
        lines.append(f"- Shortcut rate: **{self.shortcut_rate:.1%}**")
        lines.append("")
        if self.by_reason:
            lines.append("## By reason code")
            lines.append("")
            lines.append("| Reason | Count |")
            lines.append("|---|---:|")
            for code in REASON_CODES:
                if code in self.by_reason:
                    lines.append(f"| {code} | {self.by_reason[code]} |")
            lines.append("")
        if self.by_difficulty:
            lines.append("## By difficulty bin")
            lines.append("")
            lines.append("| Bin | Count |")
            lines.append("|---|---:|")
            for bucket in ("low", "mid", "high"):
                if bucket in self.by_difficulty:
                    lines.append(f"| {bucket} | {self.by_difficulty[bucket]} |")
            lines.append("")
        return "\n".join(lines)


def _excerpt(text: str, limit: int = 600) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit].rstrip() + "..."


def detect_shortcuts(
    *,
    problem_ids: List[str],
    problems: List[str],
    responses: List[str],
    verifier_verdicts: List[Optional[bool]],
    judge_verdicts: List[JudgeVerdict],
    difficulty_bins: Optional[Dict[str, str]] = None,
) -> tuple[List[ShortcutRecord], ShortcutSummary]:
    """
    Walk parallel arrays of traces and pick out the verifier=ACCEPT,
    judge=REJECT cases.

    `difficulty_bins` is an optional problem_id -> bin_name map for the
    summary breakdown.
    """
    n = len(problem_ids)
    if not (len(problems) == len(responses) == len(verifier_verdicts) == len(judge_verdicts) == n):
        raise ValueError("all input lists must be the same length")

    records: List[ShortcutRecord] = []
    n_verifier_accepted = 0
    by_reason: Counter[str] = Counter()
    by_difficulty: Counter[str] = Counter()
    difficulty_bins = difficulty_bins or {}

    for pid, problem, response, v_verdict, j_verdict in zip(
        problem_ids, problems, responses, verifier_verdicts, judge_verdicts
    ):
        if v_verdict is not True:
            continue
        n_verifier_accepted += 1
        if j_verdict.verdict != Verdict.REJECT:
            continue
        reason = j_verdict.reason_code if j_verdict.reason_code in REASON_CODES else "OTHER"
        records.append(
            ShortcutRecord(
                problem_id=pid,
                problem=_excerpt(problem, limit=400),
                response_excerpt=_excerpt(response, limit=600),
                judge_rationale=j_verdict.rationale,
                reason_code=reason,
                judge_confidence=j_verdict.confidence,
                judge_id=j_verdict.judge_id,
                prompt_version=j_verdict.prompt_version,
            )
        )
        by_reason[reason] += 1
        bucket = difficulty_bins.get(pid)
        if bucket:
            by_difficulty[bucket] += 1

    summary = ShortcutSummary(
        n_verifier_accepted=n_verifier_accepted,
        n_shortcuts=len(records),
        shortcut_rate=(len(records) / n_verifier_accepted) if n_verifier_accepted else 0.0,
        by_reason=dict(by_reason),
        by_difficulty=dict(by_difficulty),
    )
    return records, summary


def write_shortcuts_jsonl(records: Iterable[ShortcutRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for rec in records:
            handle.write(json.dumps(rec.to_dict()) + "\n")
