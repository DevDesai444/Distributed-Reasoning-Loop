"""Shortcut-detector behaviour."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation.judges import JudgeVerdict, Verdict
from evaluation.shortcut_detector import detect_shortcuts, write_shortcuts_jsonl


def _judge(verdict, reason="OTHER", rationale="x", conf=0.5):
    return JudgeVerdict(
        verdict=verdict,
        confidence=conf,
        rationale=rationale,
        reason_code=reason if verdict == Verdict.REJECT else "NONE",
        raw_response="",
        judge_id="t",
        prompt_version="MATH_RUBRIC_V1",
    )


def test_only_verifier_accept_judge_reject_count():
    pids = ["a", "b", "c", "d"]
    problems = ["p1", "p2", "p3", "p4"]
    responses = ["r1", "r2", "r3", "r4"]
    verifier = [True, True, False, True]
    judges = [
        _judge(Verdict.ACCEPT),                      # both agree -> not a shortcut
        _judge(Verdict.REJECT, "INCOMPLETE_REASONING"),  # shortcut
        _judge(Verdict.REJECT, "WRONG_METHOD"),          # verifier rejected -> not counted
        _judge(Verdict.REJECT, "LUCKY_GUESS"),           # shortcut
    ]

    records, summary = detect_shortcuts(
        problem_ids=pids,
        problems=problems,
        responses=responses,
        verifier_verdicts=verifier,
        judge_verdicts=judges,
    )

    assert len(records) == 2
    assert summary.n_verifier_accepted == 3
    assert summary.n_shortcuts == 2
    assert summary.shortcut_rate == 2 / 3
    assert summary.by_reason == {"INCOMPLETE_REASONING": 1, "LUCKY_GUESS": 1}


def test_uncertain_judge_not_counted_as_shortcut():
    records, summary = detect_shortcuts(
        problem_ids=["a"],
        problems=["p"],
        responses=["r"],
        verifier_verdicts=[True],
        judge_verdicts=[_judge(Verdict.UNCERTAIN)],
    )
    assert records == []
    assert summary.n_shortcuts == 0


def test_jsonl_round_trip(tmp_path):
    records, _ = detect_shortcuts(
        problem_ids=["a"],
        problems=["p"],
        responses=["r"],
        verifier_verdicts=[True],
        judge_verdicts=[_judge(Verdict.REJECT, "OTHER", "bad reasoning")],
    )
    path = tmp_path / "shortcuts.jsonl"
    write_shortcuts_jsonl(records, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["problem_id"] == "a"
    assert parsed["reason_code"] == "OTHER"


def test_difficulty_bin_breakdown():
    records, summary = detect_shortcuts(
        problem_ids=["a", "b"],
        problems=["p", "p"],
        responses=["r", "r"],
        verifier_verdicts=[True, True],
        judge_verdicts=[
            _judge(Verdict.REJECT, "OTHER"),
            _judge(Verdict.REJECT, "OTHER"),
        ],
        difficulty_bins={"a": "low", "b": "high"},
    )
    assert summary.by_difficulty == {"low": 1, "high": 1}
