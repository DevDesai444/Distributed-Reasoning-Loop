#!/usr/bin/env python3
"""
Pipeline smoke test for the three-way agreement harness.

This script does not benchmark anything. It exists to prove the new
verifier/RM/judge agreement code path runs end-to-end on a machine with no
GPU and no network. Real benchmark runs go through `python main.py eval
agreement ...`.

What this does:
  - Builds 5 small GSM8K-style problems inline (so no dataset download).
  - Crafts pre-known policy "traces" — some correct, some wrong, one of which
    is a known shortcut (right answer, bad reasoning).
  - Runs the math verifier on each trace.
  - Uses the stub-backend judge with a deterministic, rule-based fake LM that
    grades based on simple heuristics. This is enough to exercise the prompt
    builder, response parser, agreement metric layer, confusion-matrix
    writer, and shortcut detector.

Output: `agreement_smoke/` directory with `agreement_results.json`,
`shortcuts.jsonl`, `confusion_matrices/`, and a markdown report.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator.dataset_loader import Problem  # type: ignore
from evaluation.agreement import (
    AgreementReport,
    bin_for_problem,
    build_pair_report,
)
from evaluation.judges import JudgeConfig, SingleModelJudge
from evaluation.shortcut_detector import detect_shortcuts, write_shortcuts_jsonl
from verifier import GSM8KVerifier  # type: ignore


# ---- synthetic problems ---------------------------------------------------

SYNTH_PROBLEMS = [
    {
        "id": "smoke_1",
        "problem": "Jane has 3 apples and buys 2 more. How many does she have?",
        "answer": "5",
        # full_solution drives the difficulty binner; step count <<>>
        "full_solution": "She starts with 3 and buys 2. <<3+2=5>> #### 5",
    },
    {
        "id": "smoke_2",
        "problem": "A train travels 60 mph for 2 hours. Distance covered?",
        "answer": "120",
        "full_solution": "distance = rate * time. <<60*2=120>> #### 120",
    },
    {
        "id": "smoke_3",
        "problem": "A class has 24 students. One-third are absent. How many are present?",
        "answer": "16",
        "full_solution": "absent = <<24/3=8>>. present = <<24-8=16>>. #### 16",
    },
    {
        "id": "smoke_4",
        "problem": "Sam earns $12/hr and works 5 hours, then spends $20. How much does he have left?",
        "answer": "40",
        "full_solution": "earned = <<12*5=60>>. left = <<60-20=40>>. #### 40",
    },
    {
        "id": "smoke_5",
        "problem": "A box has 10 red and 6 blue balls. What fraction is red?",
        "answer": "10/16",
        # 4 reasoning steps so it's mid bucket
        "full_solution": "total = <<10+6=16>>. red = 10. fraction = <<10/16=0.625>>. simplify <<10/16=5/8>>. #### 10/16",
    },
]


# Each trace has a known intended verdict from the verifier + judge so the
# pipeline output can be sanity checked by eye.
SYNTH_TRACES = [
    {
        # correct, well-reasoned -> verifier accept, judge accept
        "problem_id": "smoke_1",
        "reasoning": "Step 1: she had 3. Step 2: she bought 2 more. Step 3: 3+2 = 5. #### 5",
        "final_answer": "5",
    },
    {
        # right answer but no reasoning -> shortcut candidate
        "problem_id": "smoke_2",
        "reasoning": "The answer is 120. #### 120",
        "final_answer": "120",
    },
    {
        # correct full reasoning
        "problem_id": "smoke_3",
        "reasoning": "absent = 24/3 = 8. present = 24-8 = 16. #### 16",
        "final_answer": "16",
    },
    {
        # wrong answer
        "problem_id": "smoke_4",
        "reasoning": "He earns 12*5=60 then spends 20, so 60+20=80. #### 80",
        "final_answer": "80",
    },
    {
        # wrong answer with confused reasoning
        "problem_id": "smoke_5",
        "reasoning": "There are 6 red, so 6/16. #### 6/16",
        "final_answer": "6/16",
    },
]


# Heuristic stub LLM. It looks at the response text and decides:
#  - if response has the word "step" plus a digit-arithmetic pattern, ACCEPT.
#  - if response is just "The answer is X" with no derivation, REJECT with
#    LUCKY_GUESS.
#  - if the answer looks numerically wrong vs the reference, REJECT with
#    WRONG_METHOD.
#  - otherwise ACCEPT.
#
# Crucially, the judge sees the reference answer in the rubric prompt, so
# we can extract it.
def _stub_judge_fn(prompt: str) -> str:
    # extract reference answer
    ref_match = re.search(r"Reference answer.*?:\s*(.+?)\n", prompt, flags=re.DOTALL)
    ref = ref_match.group(1).strip() if ref_match else ""
    # extract student response
    resp_match = re.search(r"Student solution:\s*(.*?)\nReference answer", prompt, flags=re.DOTALL)
    response = resp_match.group(1).strip() if resp_match else ""

    def _digits(s: str) -> Optional[str]:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
        return m[-1] if m else None

    ref_num = _digits(ref)
    resp_num = _digits(response)

    # right-answer-only patterns
    no_derivation = re.search(r"^\s*(?:the\s+)?answer\s+is\s+", response, flags=re.IGNORECASE) is not None
    has_steps = ("step" in response.lower()) or ("=" in response and len(response.split("=")) >= 3)

    if ref_num and resp_num and ref_num == resp_num and no_derivation and not has_steps:
        return (
            "VERDICT: REJECT\n"
            "CONFIDENCE: 0.8\n"
            "REASON_CODE: LUCKY_GUESS\n"
            "RATIONALE: final number matches but no derivation shown."
        )
    if ref_num and resp_num and ref_num != resp_num:
        return (
            "VERDICT: REJECT\n"
            "CONFIDENCE: 0.7\n"
            "REASON_CODE: WRONG_METHOD\n"
            "RATIONALE: final answer does not match expected."
        )
    if has_steps:
        return (
            "VERDICT: ACCEPT\n"
            "CONFIDENCE: 0.85\n"
            "REASON_CODE: NONE\n"
            "RATIONALE: steps shown and arithmetic checks out."
        )
    return (
        "VERDICT: UNCERTAIN\n"
        "CONFIDENCE: 0.3\n"
        "REASON_CODE: OTHER\n"
        "RATIONALE: hard to tell from response alone."
    )


def main() -> int:
    # samples/ is a checked-in directory so the artifact lives with the repo
    # and reviewers can see the format without re-running the script.
    out_dir = Path("./samples/agreement_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = [
        Problem(
            id=p["id"],
            problem=p["problem"],
            answer=p["answer"],
            metadata={"full_solution": p["full_solution"]},
        )
        for p in SYNTH_PROBLEMS
    ]
    problems_by_id = {p.id: p for p in problems}

    # verifier
    verifier = GSM8KVerifier()
    verifier_verdicts: List[Optional[bool]] = []
    for trace in SYNTH_TRACES:
        problem = problems_by_id[trace["problem_id"]]
        result = verifier.verify_reasoning_path(trace["reasoning"], problem.answer)
        if result.status.value == "correct":
            verifier_verdicts.append(True)
        elif result.status.value == "incorrect":
            verifier_verdicts.append(False)
        else:
            verifier_verdicts.append(None)

    # judge — stubbed deterministic fake
    judge = SingleModelJudge(JudgeConfig(model_name="stub-judge", backend="stub"))
    judge.set_stub(_stub_judge_fn)

    judge_verdicts = []
    for trace in SYNTH_TRACES:
        problem = problems_by_id[trace["problem_id"]]
        v = judge.judge(
            problem=problem.problem,
            response=trace["reasoning"],
            reference_answer=problem.answer,
            problem_type="math",
        )
        judge_verdicts.append(v)
    judge_bool: List[Optional[bool]] = [v.as_bool for v in judge_verdicts]

    # difficulty bins
    pid_to_bucket = {p.id: bin_for_problem(p) for p in problems}
    indices_by_bin = {"low": [], "mid": [], "high": []}
    for i, trace in enumerate(SYNTH_TRACES):
        bucket = pid_to_bucket.get(trace["problem_id"], "low")
        indices_by_bin[bucket].append(i)

    pair_reports = [
        build_pair_report(
            "verifier", "judge",
            verifier_verdicts, judge_bool,
            bins=indices_by_bin,
            n_bootstrap=200,
        ),
    ]

    report = AgreementReport(
        run_id="agreement_smoke",
        model="(synthetic)",
        benchmark="gsm8k-synthetic",
        n_problems=len(problems),
        n_traces=len(SYNTH_TRACES),
        judge_model="stub-judge",
        judge_prompt_version=judge_verdicts[0].prompt_version,
        pair_reports=pair_reports,
        metadata={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "note": "pipeline smoke artifact — not a benchmark claim",
            "judge_backend": "stub",
        },
    )

    (out_dir / "agreement_smoke.json").write_text(report.render_json())
    (out_dir / "report.md").write_text(report.render_markdown())

    # confusion matrices
    confusion_dir = out_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for pair in pair_reports:
        (confusion_dir / f"{pair.rater_a}_vs_{pair.rater_b}.json").write_text(
            json.dumps(pair.confusion, indent=2)
        )

    # shortcuts
    records, summary = detect_shortcuts(
        problem_ids=[t["problem_id"] for t in SYNTH_TRACES],
        problems=[problems_by_id[t["problem_id"]].problem for t in SYNTH_TRACES],
        responses=[t["reasoning"] for t in SYNTH_TRACES],
        verifier_verdicts=verifier_verdicts,
        judge_verdicts=judge_verdicts,
        difficulty_bins=pid_to_bucket,
    )
    write_shortcuts_jsonl(records, out_dir / "shortcuts.jsonl")
    (out_dir / "shortcuts_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2)
    )

    print(f"smoke run wrote artifacts to {out_dir}")
    print(f"verifier accepts: {sum(1 for v in verifier_verdicts if v)}")
    print(f"judge accepts:    {sum(1 for v in judge_bool if v)}")
    print(f"shortcuts:        {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
