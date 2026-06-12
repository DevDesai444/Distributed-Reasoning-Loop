#!/usr/bin/env python3
"""
Three-way agreement evaluation runner.

Generates traces with a policy model, then scores each trace with:
  1. the existing math / code verifier
  2. the existing outcome reward model (binary-thresholded)
  3. an LLM-as-judge

and emits agreement metrics, confusion matrices, and a shortcut report.

This is the entry point behind `python main.py eval agreement ...`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Lazy heavy imports: we want `--help` to be fast on machines without torch.

def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("agreement_%Y%m%dT%H%M%SZ")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run three-way evaluator agreement on a benchmark slice.",
    )
    p.add_argument("--model", type=str, required=True,
                   help="Policy model under test (HF id or local path).")
    p.add_argument("--benchmark", type=str, default="gsm8k",
                   choices=["gsm8k", "humaneval", "math"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--n-problems", type=int, default=20,
                   help="Number of problems to sample from the benchmark.")
    p.add_argument("--n-samples", type=int, default=1,
                   help="Number of traces to generate per problem.")
    p.add_argument("--judge-model", type=str,
                   default="Qwen/Qwen2.5-7B-Instruct",
                   help="Open-weights model used as the LLM judge.")
    p.add_argument("--judge-backend", type=str, default="auto",
                   choices=["auto", "vllm", "transformers", "stub"])
    p.add_argument("--judge-max-tokens", type=int, default=256)
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--reward-model", type=str, default=None,
                   help="Path to a trained reward-model checkpoint. "
                        "If omitted, RM column is skipped.")
    p.add_argument("--rm-threshold", type=float, default=None,
                   help="Reward score threshold for binary verdict. "
                        "Default is the median over the run.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where artifacts go. Default: ./outputs/<run_id>/")
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--position-swap", action="store_true",
                   help="Run a pairwise position-swap probe alongside the main eval.")
    p.add_argument("--self-preference", action="store_true",
                   help="Run a self-preference probe when judge model == policy model.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--problem-type", type=str, default=None,
                   help="Override the rubric problem type (math|code). "
                        "Default derived from benchmark.")
    p.add_argument("--policy-traces", type=str, default=None,
                   help="Optional JSONL of pre-generated traces "
                        "(skips policy generation).")
    return p


def _benchmark_to_problem_type(benchmark: str) -> str:
    if benchmark == "humaneval":
        return "code"
    return "math"


def _load_problems(benchmark: str, split: str, n: int):
    from data_generator import GSM8KLoader, HumanEvalLoader
    if benchmark == "gsm8k":
        loader = GSM8KLoader(split=split, subset_size=n)
    elif benchmark == "humaneval":
        loader = HumanEvalLoader(subset_size=n)
    elif benchmark == "math":
        try:
            from data_generator.dataset_loader import MATHLoader  # type: ignore
        except ImportError as exc:
            raise SystemExit(f"MATH loader not available: {exc}")
        loader = MATHLoader(split=split, subset_size=n)  # type: ignore[call-arg]
    else:
        raise ValueError(f"Unsupported benchmark {benchmark!r}")
    return loader.load()


def _generate_traces(
    *,
    model_name: str,
    problems,
    n_samples: int,
    problem_type: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Returns a flat list of {problem_id, problem_idx, sample_idx, reasoning,
    final_answer} dicts. Uses CoTGenerator with whichever backend is
    available; falls back to transformers.
    """
    from data_generator import CoTGenerator, GenerationConfig
    from data_generator.cot_generator import InferenceBackend

    # try vLLM if importable, else fall back
    try:
        import vllm  # noqa: F401
        backend = InferenceBackend.VLLM
    except Exception:
        backend = InferenceBackend.TRANSFORMERS

    config = GenerationConfig(
        model_name=model_name,
        backend=backend,
        num_paths=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    gen = CoTGenerator(config)
    gen.initialize()

    traces: List[Dict[str, Any]] = []
    for prob_idx, problem in enumerate(problems):
        paths = gen.generate_single(
            problem=problem.problem,
            problem_id=problem.id,
            problem_type=problem_type,
        )
        # pad/truncate to n_samples
        if not paths:
            traces.append({
                "problem_id": problem.id,
                "problem_idx": prob_idx,
                "sample_idx": 0,
                "reasoning": "",
                "final_answer": None,
            })
            continue
        for s_idx, path in enumerate(paths[:n_samples]):
            traces.append({
                "problem_id": problem.id,
                "problem_idx": prob_idx,
                "sample_idx": s_idx,
                "reasoning": path.reasoning,
                "final_answer": path.final_answer,
            })
    return traces


def _load_pregenerated(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _verifier_for(benchmark: str):
    from verifier import GSM8KVerifier, MathVerifier  # type: ignore
    if benchmark == "gsm8k":
        return GSM8KVerifier()
    if benchmark == "math":
        return MathVerifier()
    if benchmark == "humaneval":
        from verifier import HumanEvalVerifier  # type: ignore
        return HumanEvalVerifier()
    raise ValueError(benchmark)


def _verifier_verdict(verifier, benchmark: str, trace: Dict[str, Any], problem) -> Optional[bool]:
    from verifier import VerificationStatus  # type: ignore
    try:
        if benchmark == "humaneval":
            code = verifier.extract_code(trace["reasoning"])
            from verifier import ExecutionStatus  # type: ignore
            result = verifier.verify_problem(
                task_id=problem.id,
                prompt=problem.problem,
                completion=code,
                test=problem.metadata["test"],
                entry_point=problem.metadata["entry_point"],
            )
            return result.status == ExecutionStatus.SUCCESS
        # math-ish
        result = verifier.verify_reasoning_path(trace["reasoning"], problem.answer)
        if result.status == VerificationStatus.CORRECT:
            return True
        if result.status == VerificationStatus.INCORRECT:
            return False
        return None
    except Exception as exc:
        logger.warning("verifier crashed on %s: %s", problem.id, exc)
        return None


def _rm_scores(reward_model_path: Optional[str], traces: List[Dict[str, Any]], problems_by_id) -> List[Optional[float]]:
    if not reward_model_path:
        return [None] * len(traces)
    from training.reward_model import RewardModel, RewardModelConfig  # type: ignore

    rm = RewardModel(RewardModelConfig(model_name=reward_model_path))
    try:
        rm.load(reward_model_path)
    except Exception as exc:
        logger.warning("could not load RM from %s: %s — skipping RM column", reward_model_path, exc)
        return [None] * len(traces)

    scores: List[Optional[float]] = []
    for trace in traces:
        problem = problems_by_id[trace["problem_id"]]
        try:
            score = rm.compute_reward(problem.problem, trace["reasoning"])
            scores.append(float(score))
        except Exception as exc:
            logger.warning("RM scoring failed on %s: %s", trace["problem_id"], exc)
            scores.append(None)
    return scores


def _rm_verdicts(scores: List[Optional[float]], threshold: Optional[float]) -> Tuple[List[Optional[bool]], Optional[float]]:
    valid = [s for s in scores if s is not None]
    if not valid:
        return [None] * len(scores), None
    if threshold is None:
        # median
        sorted_v = sorted(valid)
        mid = len(sorted_v) // 2
        threshold = (sorted_v[mid] if len(sorted_v) % 2 else (sorted_v[mid - 1] + sorted_v[mid]) / 2)
    out: List[Optional[bool]] = []
    for s in scores:
        if s is None:
            out.append(None)
        else:
            out.append(s >= threshold)
    return out, threshold


def _judge_verdicts(
    *,
    judge_model: str,
    judge_backend: str,
    judge_max_tokens: int,
    judge_temperature: float,
    traces: List[Dict[str, Any]],
    problems_by_id,
    problem_type: str,
) -> Tuple[List, Any]:
    from evaluation.judges import JudgeConfig, SingleModelJudge

    cfg = JudgeConfig(
        model_name=judge_model,
        backend=judge_backend,
        max_new_tokens=judge_max_tokens,
        temperature=judge_temperature,
    )
    judge = SingleModelJudge(cfg)

    verdicts = []
    for trace in traces:
        problem = problems_by_id[trace["problem_id"]]
        v = judge.judge(
            problem=problem.problem,
            response=trace["reasoning"],
            reference_answer=problem.answer,
            problem_type=problem_type,
        )
        verdicts.append(v)
    return verdicts, judge


def _build_bins(problems, traces):
    from evaluation.agreement import bin_for_problem
    pid_to_bucket = {p.id: bin_for_problem(p) for p in problems}
    indices_by_bin: Dict[str, List[int]] = {"low": [], "mid": [], "high": []}
    for i, trace in enumerate(traces):
        bucket = pid_to_bucket.get(trace["problem_id"], "low")
        indices_by_bin[bucket].append(i)
    return pid_to_bucket, indices_by_bin


def _write_confusion(out_dir: Path, name: str, matrix: Dict[str, int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    md_path = out_dir / f"{name}.md"
    with open(json_path, "w") as fh:
        json.dump(matrix, fh, indent=2)
    md_lines = [
        f"# Confusion — {name}",
        "",
        "| | B accept | B reject |",
        "|---|---:|---:|",
        f"| A accept | {matrix['tp']} | {matrix['fn']} |",
        f"| A reject | {matrix['fp']} | {matrix['tn']} |",
        f"(dropped uncertain pairs: {matrix.get('dropped_uncertain', 0)})",
    ]
    md_path.write_text("\n".join(md_lines))


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    run_id = _now_run_id()
    out_dir = Path(args.output_dir) if args.output_dir else Path("./outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    problem_type = args.problem_type or _benchmark_to_problem_type(args.benchmark)

    logger.info("Loading %s problems from %s", args.n_problems, args.benchmark)
    problems = _load_problems(args.benchmark, args.split, args.n_problems)
    problems_by_id = {p.id: p for p in problems}

    if args.policy_traces:
        logger.info("Loading pre-generated traces from %s", args.policy_traces)
        traces = _load_pregenerated(args.policy_traces)
    else:
        logger.info("Generating %s sample(s) per problem from %s", args.n_samples, args.model)
        traces = _generate_traces(
            model_name=args.model,
            problems=problems,
            n_samples=args.n_samples,
            problem_type=problem_type,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )

    # ---- verifier verdicts ---------------------------------------------
    verifier = _verifier_for(args.benchmark)
    verifier_verdicts: List[Optional[bool]] = []
    for trace in traces:
        problem = problems_by_id[trace["problem_id"]]
        verifier_verdicts.append(_verifier_verdict(verifier, args.benchmark, trace, problem))

    # ---- reward-model verdicts -----------------------------------------
    rm_scores = _rm_scores(args.reward_model, traces, problems_by_id)
    rm_verdicts, rm_threshold = _rm_verdicts(rm_scores, args.rm_threshold)

    # ---- judge verdicts ------------------------------------------------
    logger.info("Running judge: %s (backend=%s)", args.judge_model, args.judge_backend)
    judge_verdicts, judge = _judge_verdicts(
        judge_model=args.judge_model,
        judge_backend=args.judge_backend,
        judge_max_tokens=args.judge_max_tokens,
        judge_temperature=args.judge_temperature,
        traces=traces,
        problems_by_id=problems_by_id,
        problem_type=problem_type,
    )
    judge_bool: List[Optional[bool]] = [v.as_bool for v in judge_verdicts]

    # ---- per-trace artifact --------------------------------------------
    per_trace = []
    for trace, v_verdict, r_verdict, r_score, j_verdict in zip(
        traces, verifier_verdicts, rm_verdicts, rm_scores, judge_verdicts
    ):
        per_trace.append({
            "problem_id": trace["problem_id"],
            "sample_idx": trace["sample_idx"],
            "reasoning_excerpt": (trace["reasoning"] or "")[:600],
            "final_answer": trace.get("final_answer"),
            "verifier_verdict": v_verdict,
            "rm_score": r_score,
            "rm_verdict": r_verdict,
            "judge": j_verdict.to_dict(),
        })
    with open(out_dir / "per_trace.jsonl", "w") as fh:
        for row in per_trace:
            fh.write(json.dumps(row) + "\n")

    # ---- agreement / confusion / bins ----------------------------------
    from evaluation.agreement import (
        AgreementReport,
        build_pair_report,
        confusion_matrix,
    )

    _, indices_by_bin = _build_bins(problems, traces)

    pair_reports = []
    pair_reports.append(build_pair_report(
        "verifier", "judge",
        verifier_verdicts, judge_bool,
        bins=indices_by_bin, n_bootstrap=args.n_bootstrap,
    ))
    if any(v is not None for v in rm_verdicts):
        pair_reports.append(build_pair_report(
            "verifier", "reward_model",
            verifier_verdicts, rm_verdicts,
            bins=indices_by_bin, n_bootstrap=args.n_bootstrap,
        ))
        pair_reports.append(build_pair_report(
            "reward_model", "judge",
            rm_verdicts, judge_bool,
            bins=indices_by_bin, n_bootstrap=args.n_bootstrap,
        ))

    judge_prompt_version = judge_verdicts[0].prompt_version if judge_verdicts else ""
    report = AgreementReport(
        run_id=run_id,
        model=args.model,
        benchmark=args.benchmark,
        n_problems=len(problems),
        n_traces=len(traces),
        judge_model=args.judge_model,
        judge_prompt_version=judge_prompt_version,
        pair_reports=pair_reports,
        metadata={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "n_samples": args.n_samples,
            "judge_backend": args.judge_backend,
            "rm_threshold": rm_threshold,
            "policy_max_new_tokens": args.max_new_tokens,
            "policy_temperature": args.temperature,
            "seed": args.seed,
            "problem_type": problem_type,
        },
    )

    # write results
    (out_dir / "agreement_results.json").write_text(report.render_json())
    (out_dir / "report.md").write_text(report.render_markdown())

    confusion_dir = out_dir / "confusion_matrices"
    for pair in pair_reports:
        name = f"{pair.rater_a}_vs_{pair.rater_b}"
        _write_confusion(confusion_dir, name, pair.confusion)

    # ---- shortcut detection --------------------------------------------
    from evaluation.shortcut_detector import detect_shortcuts, write_shortcuts_jsonl
    pid_to_bucket = {p.id: __import__("evaluation.agreement", fromlist=["bin_for_problem"]).bin_for_problem(p) for p in problems}
    records, summary = detect_shortcuts(
        problem_ids=[t["problem_id"] for t in traces],
        problems=[problems_by_id[t["problem_id"]].problem for t in traces],
        responses=[t["reasoning"] for t in traces],
        verifier_verdicts=verifier_verdicts,
        judge_verdicts=judge_verdicts,
        difficulty_bins=pid_to_bucket,
    )
    write_shortcuts_jsonl(records, out_dir / "shortcuts.jsonl")
    (out_dir / "shortcuts_summary.md").write_text(
        summary.render_markdown(title=f"Shortcuts — {args.benchmark}")
    )
    with open(out_dir / "shortcuts_summary.json", "w") as fh:
        json.dump(summary.to_dict(), fh, indent=2)

    # ---- run manifest --------------------------------------------------
    manifest = {
        "run_id": run_id,
        "kind": "agreement",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "judge_model": args.judge_model,
        "judge_prompt_version": judge_prompt_version,
        "benchmark": args.benchmark,
        "split": args.split,
        "n_problems": len(problems),
        "n_traces": len(traces),
        "rm_threshold": rm_threshold,
        "artifacts": [
            "agreement_results.json",
            "report.md",
            "confusion_matrices/",
            "shortcuts.jsonl",
            "shortcuts_summary.md",
            "shortcuts_summary.json",
            "per_trace.jsonl",
        ],
    }
    with open(out_dir / "run_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info("Agreement run complete -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
