#!/usr/bin/env python3
"""
Robustness evaluation runner.

Loads a benchmark slice, runs baseline + six perturbations, and reports
per-perturbation pass@1 retention with bootstrap CIs and an overall
geometric-mean robustness score.

This is the entry point behind `python main.py eval robust ...`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("robustness_%Y%m%dT%H%M%SZ")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run robustness evaluation on a benchmark slice.",
    )
    p.add_argument("--model", type=str, required=True,
                   help="Policy model (HF id or local path).")
    p.add_argument("--benchmark", type=str, default="gsm8k",
                   choices=["gsm8k"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--n-problems", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where artifacts go. Default ./outputs/<run_id>/")
    p.add_argument("--policy-traces", type=str, default=None,
                   help="Optional JSONL with pre-generated traces "
                        "(skips model loading entirely).")
    return p


def _load_problems(benchmark: str, split: str, n: int):
    from data_generator import GSM8KLoader
    if benchmark != "gsm8k":
        raise SystemExit(f"unsupported benchmark for robust: {benchmark!r}")
    return GSM8KLoader(split=split, subset_size=n).load()


def _verifier_for(benchmark: str):
    from verifier import GSM8KVerifier  # type: ignore
    if benchmark == "gsm8k":
        return GSM8KVerifier()
    raise ValueError(benchmark)


def _make_verify_fn(benchmark: str) -> Callable[[str, str], Optional[bool]]:
    from verifier import VerificationStatus  # type: ignore
    verifier = _verifier_for(benchmark)

    def verify(output: str, gold: str) -> Optional[bool]:
        try:
            result = verifier.verify_reasoning_path(output, gold)
            if result.status == VerificationStatus.CORRECT:
                return True
            if result.status == VerificationStatus.INCORRECT:
                return False
            return None
        except Exception as exc:
            logger.warning("verifier crashed: %s", exc)
            return None
    return verify


def _make_generate_fn(
    *,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    pregenerated_path: Optional[str],
) -> Callable[[str, int], List[str]]:
    if pregenerated_path:
        # pregenerated traces: dict keyed by exact problem text -> list[str]
        with open(pregenerated_path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        lookup: Dict[str, List[str]] = {}
        for r in rows:
            lookup.setdefault(r["problem"], []).append(r["reasoning"])

        def generate(problem: str, n: int) -> List[str]:
            entries = lookup.get(problem, [])
            if not entries:
                return [""] * n
            # repeat-cycle to n
            out = []
            for i in range(n):
                out.append(entries[i % len(entries)])
            return out
        return generate

    # Live generation via CoTGenerator (same as agreement_eval).
    from data_generator import CoTGenerator, GenerationConfig
    from data_generator.cot_generator import InferenceBackend

    try:
        import vllm  # noqa: F401
        backend = InferenceBackend.VLLM
    except Exception:
        backend = InferenceBackend.TRANSFORMERS

    config = GenerationConfig(
        model_name=model_name,
        backend=backend,
        num_paths=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    gen = CoTGenerator(config)
    gen.initialize()

    def generate(problem: str, n: int) -> List[str]:
        # Reuse CoTGenerator for n samples by calling repeatedly. Most
        # backends will batch this for us; a tighter integration is a
        # follow-up.
        outputs: List[str] = []
        for _ in range(n):
            paths = gen.generate_single(
                problem=problem,
                problem_id="adhoc",
                problem_type="math",
            )
            if paths:
                outputs.append(paths[0].reasoning)
            else:
                outputs.append("")
        return outputs

    return generate


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    run_id = _now_run_id()
    out_dir = Path(args.output_dir) if args.output_dir else Path("./outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s problems from %s", args.n_problems, args.benchmark)
    problems = _load_problems(args.benchmark, args.split, args.n_problems)
    if not problems:
        logger.error("no problems loaded")
        return 2

    verify_fn = _make_verify_fn(args.benchmark)
    generate_fn = _make_generate_fn(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        pregenerated_path=args.policy_traces,
    )

    from evaluation.robustness import default_perturbations, evaluate_robustness

    perturbations = default_perturbations()
    report = evaluate_robustness(
        problems=problems,
        n_samples=args.n_samples,
        perturbations=perturbations,
        generate_fn=generate_fn,
        verify_fn=verify_fn,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        model_label=args.model,
        benchmark_label=args.benchmark,
        run_id=run_id,
        extra_metadata={
            "split": args.split,
            "n_samples_per_problem": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
    )

    (out_dir / "robustness_results.json").write_text(report.render_json())
    (out_dir / "report.md").write_text(report.render_markdown())

    manifest = {
        "run_id": run_id,
        "kind": "robustness",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "benchmark": args.benchmark,
        "split": args.split,
        "n_problems": len(problems),
        "n_samples_per_problem": args.n_samples,
        "seed": args.seed,
        "overall_robustness": report.overall_robustness,
        "artifacts": [
            "robustness_results.json",
            "report.md",
        ],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Robustness run complete -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
