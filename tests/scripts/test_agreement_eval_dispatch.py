"""Smoke tests for the agreement runner's benchmark dispatch.

These exercise the math and humaneval code paths end-to-end without a
GPU, without Docker, and without an HF dataset download. We stub the
loader and the policy generator, then assert the runner emits the
expected agreement_results.json shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _math_problems() -> List[Any]:
    from data_generator.dataset_loader import Problem  # type: ignore
    return [
        Problem(
            id="math_smoke_1",
            problem="What is 2 + 3?",
            answer="5",
            metadata={"level": "Level 1", "full_solution": "<<2+3=5>> #### 5"},
        ),
        Problem(
            id="math_smoke_2",
            problem="What is 4 * 6?",
            answer="24",
            metadata={"level": "Level 4", "full_solution": "<<4*6=24>> #### 24"},
        ),
    ]


def _humaneval_problems() -> List[Any]:
    from data_generator.dataset_loader import Problem  # type: ignore
    return [
        Problem(
            id="he_smoke_1",
            problem="def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n",
            answer="    return a + b\n",
            metadata={
                "entry_point": "add",
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate(1, 1) == 2\n"
                ),
            },
        ),
    ]


def _stub_judge_response(prompt: str) -> str:
    # Deterministic, generic ACCEPT so we can verify the pipeline plumbs
    # the rubric through correctly regardless of benchmark.
    return (
        "VERDICT: ACCEPT\n"
        "CONFIDENCE: 0.7\n"
        "REASON_CODE: NONE\n"
        "RATIONALE: smoke stub accepts everything.\n"
    )


def _patch_loaders(monkeypatch, benchmark: str, problems: List[Any]) -> None:
    from scripts import agreement_eval

    def _fake_load_problems(b, split, n):
        assert b == benchmark
        return problems[:n] if n is not None else problems

    monkeypatch.setattr(agreement_eval, "_load_problems", _fake_load_problems)


def _patch_generator(monkeypatch, traces_by_id: Dict[str, str]) -> None:
    from scripts import agreement_eval

    def _fake_generate(*, model_name, problems, n_samples, problem_type,
                       max_new_tokens, temperature, seed):
        traces = []
        for idx, problem in enumerate(problems):
            traces.append({
                "problem_id": problem.id,
                "problem_idx": idx,
                "sample_idx": 0,
                "reasoning": traces_by_id.get(problem.id, ""),
                "final_answer": None,
            })
        return traces

    monkeypatch.setattr(agreement_eval, "_generate_traces", _fake_generate)


def _patch_judge(monkeypatch) -> None:
    from scripts import agreement_eval
    from evaluation.judges import JudgeConfig, SingleModelJudge

    def _fake_judge_verdicts(*, judge_model, judge_backend, judge_max_tokens,
                              judge_temperature, traces, problems_by_id, problem_type):
        judge = SingleModelJudge(JudgeConfig(model_name="stub", backend="stub"))
        judge.set_stub(lambda prompt: _stub_judge_response(prompt))
        verdicts = []
        for trace in traces:
            problem = problems_by_id[trace["problem_id"]]
            verdicts.append(judge.judge(
                problem=problem.problem,
                response=trace["reasoning"],
                reference_answer=problem.answer,
                problem_type=problem_type,
            ))
        return verdicts, judge

    monkeypatch.setattr(agreement_eval, "_judge_verdicts", _fake_judge_verdicts)


def test_cli_accepts_math_and_humaneval():
    from scripts.agreement_eval import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "--model", "stub",
        "--benchmark", "math",
        "--judge-model", "stub",
    ])
    assert args.benchmark == "math"

    args2 = parser.parse_args([
        "--model", "stub",
        "--benchmark", "humaneval",
        "--judge-model", "stub",
    ])
    assert args2.benchmark == "humaneval"


def test_math_dispatch_end_to_end(tmp_path, monkeypatch):
    from scripts.agreement_eval import main as run

    problems = _math_problems()
    _patch_loaders(monkeypatch, "math", problems)
    _patch_generator(monkeypatch, {
        "math_smoke_1": "Step 1: 2+3 = 5. #### 5",
        "math_smoke_2": "Step 1: 4*6 = 24. #### 24",
    })
    _patch_judge(monkeypatch)

    out_dir = tmp_path / "math_run"
    rc = run([
        "--model", "stub",
        "--benchmark", "math",
        "--judge-model", "stub",
        "--judge-backend", "stub",
        "--n-problems", "2",
        "--n-samples", "1",
        "--n-bootstrap", "20",
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    results = json.loads((out_dir / "agreement_results.json").read_text())
    assert results["benchmark"] == "math"
    assert results["n_traces"] == 2
    # Bins should be MATH levels, not GSM8K low/mid/high.
    per_bin = results["pair_reports"][0]["per_bin"]
    assert any(key.startswith("level_") for key in per_bin), per_bin


def test_humaneval_dispatch_end_to_end(tmp_path, monkeypatch):
    from scripts.agreement_eval import main as run

    problems = _humaneval_problems()
    _patch_loaders(monkeypatch, "humaneval", problems)
    _patch_generator(monkeypatch, {
        "he_smoke_1": "```python\n    return a + b\n```",
    })
    _patch_judge(monkeypatch)

    out_dir = tmp_path / "he_run"
    rc = run([
        "--model", "stub",
        "--benchmark", "humaneval",
        "--judge-model", "stub",
        "--judge-backend", "stub",
        "--n-problems", "1",
        "--n-samples", "1",
        "--n-bootstrap", "20",
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    results = json.loads((out_dir / "agreement_results.json").read_text())
    assert results["benchmark"] == "humaneval"
    assert results["metadata"]["problem_type"] == "code"
    # Single "all" bucket.
    per_bin = results["pair_reports"][0]["per_bin"]
    assert list(per_bin.keys()) == ["all"]
