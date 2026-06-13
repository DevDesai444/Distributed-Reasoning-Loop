"""
Bias-control behaviour for judge subsystem.

We don't need a real LLM here — the SingleModelJudge supports a stub backend
that returns canned responses, and the bias-control utilities are pure
functions on verdict sequences.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation.judges import (
    JudgeConfig,
    SingleModelJudge,
    Verdict,
    length_bias_probe,
    position_swap_stability,
    self_preference_delta,
    truncate_to_min_length,
)
from evaluation.judges.pairwise_judge import PairwiseJudge


def test_position_swap_all_stable():
    forward = ["A", "B", "TIE"]
    reverse = ["B", "A", "TIE"]  # A in fwd = B in rev, B = A, TIE = TIE
    result = position_swap_stability(forward, reverse)
    assert result.n_stable == 3
    assert result.swap_rate == 0.0


def test_position_swap_all_unstable():
    forward = ["A", "B"]
    reverse = ["A", "B"]  # A=A (unstable), B=B (unstable)
    result = position_swap_stability(forward, reverse)
    assert result.n_stable == 0
    assert result.swap_rate == 1.0


def test_truncate_to_min_length():
    a, b, c = truncate_to_min_length("aaaaa", "bb", "ccc")
    assert len(a) == len(b) == len(c) == 2


def test_self_preference_delta():
    self_v = [Verdict.ACCEPT] * 8 + [Verdict.REJECT] * 2
    other_v = [Verdict.ACCEPT] * 4 + [Verdict.REJECT] * 6
    result = self_preference_delta(self_v, other_v)
    assert result.n_problems == 10
    assert result.self_accept_rate == 0.8
    assert result.other_accept_rate == 0.4
    assert abs(result.delta - 0.4) < 1e-9


def test_length_bias_probe_via_stub_judge():
    # A judge that just inspects whether the response is longer than 50 chars
    judge = SingleModelJudge(JudgeConfig(backend="stub", model_name="stub-judge"))

    def fake_gen(prompt: str) -> str:
        # find the response section
        if "long" in prompt and "[...truncated]" not in prompt:
            # response contains the long-wrong text
            return "VERDICT: ACCEPT\nCONFIDENCE: 0.7\nREASON_CODE: NONE\nRATIONALE: looks thorough"
        return "VERDICT: REJECT\nCONFIDENCE: 0.6\nREASON_CODE: INCOMPLETE_REASONING\nRATIONALE: too short"

    judge.set_stub(fake_gen)

    def probe_fn(problem, response):
        v = judge.judge(problem=problem, response=response, problem_type="math")
        return v.verdict

    result = length_bias_probe(
        probe_fn,
        long_wrong="this is a long answer that contains the word long " * 5,
        short_right="42",
    )
    # synthetic stub: long->accept, short->reject -> bias flagged
    assert result["length_bias_detected"] is True


def test_pairwise_swap_unstable_downgrades_to_tie():
    judge = SingleModelJudge(JudgeConfig(backend="stub", model_name="stub"))

    # alternate so forward says A wins, reverse also says A (unstable)
    seen = {"calls": 0}

    def fake_gen(prompt: str) -> str:
        seen["calls"] += 1
        return "WINNER: A\nCONFIDENCE: 0.8\nRATIONALE: chose first"

    judge.set_stub(fake_gen)
    pair = PairwiseJudge(judge)

    verdict = pair.judge_pair(
        problem="p", response_a="aaa", response_b="bbb", check_position_swap=True
    )
    assert verdict.position_swap_stable is False
    assert verdict.winner == "TIE"
    assert seen["calls"] == 2


def test_pairwise_swap_stable_keeps_winner():
    judge = SingleModelJudge(JudgeConfig(backend="stub", model_name="stub"))

    # forward picks the first listed; reverse picks the first listed too;
    # forward calls always come first, so in reverse the original B is in
    # slot A and should be reported as the winner there
    state = {"call": 0}

    def fake_gen(prompt: str) -> str:
        state["call"] += 1
        if state["call"] == 1:
            return "WINNER: A\nCONFIDENCE: 0.9\nRATIONALE: ."
        return "WINNER: B\nCONFIDENCE: 0.7\nRATIONALE: ."

    judge.set_stub(fake_gen)
    pair = PairwiseJudge(judge)
    verdict = pair.judge_pair(
        problem="p", response_a="aaa", response_b="bbb", check_position_swap=True
    )
    assert verdict.position_swap_stable is True
    assert verdict.winner == "A"
