#!/usr/bin/env python3
"""
Script to evaluate reasoning models on benchmarks.
Supports GSM8K, HumanEval, and MATH benchmarks.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from run_artifacts import RunArtifacts
from tracks import apply_track_policy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value or None


def build_run_metadata(args: argparse.Namespace) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "benchmark": args.benchmark,
        "split": args.split,
        "subset_size": args.subset_size,
        "use_ttc": args.use_ttc,
        "ttc_samples": args.ttc_samples if args.use_ttc else None,
        "ttc_oracle_verify": args.ttc_oracle_verify,
        "output_dir": str(Path(args.output_dir).resolve()),
        "cwd": os.getcwd(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
    }


def persist_run_manifest(output_dir: str, metadata: dict) -> None:
    with open(Path(output_dir) / "run_manifest.json", "w") as f:
        json.dump(metadata, f, indent=2)


def record_benchmark_result(
    manager: RunArtifacts,
    *,
    benchmark_name: str,
    result,
    output_path: str,
) -> None:
    manager.record_metric(f"{benchmark_name}_accuracy", result.accuracy)
    manager.record_metric(f"{benchmark_name}_errors", result.errors)
    manager.record_artifact(
        stage="evaluation",
        name=f"{benchmark_name}_results",
        path=output_path,
        metadata=result.to_dict(),
    )


def summarize_suite_status(results: dict) -> str:
    statuses = [result.status for result in results.values()]
    if not statuses:
        return "empty"
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "partial" for status in statuses):
        return "partial"
    return "completed"


def finalize_result(result, output_path: str, output_dir: str, fail_on_errors: bool) -> int:
    result.save(output_path)

    if result.status == "failed":
        logger.error(
            "Run failed on every problem. Treat this artifact as invalid; "
            "check the per-problem errors in the saved JSON."
        )
        return 2

    if result.status == "partial":
        logger.warning(
            "Run completed with %s errors out of %s problems. "
            "The reported accuracy is incomplete.",
            result.errors,
            result.total_problems,
        )
        if fail_on_errors:
            return 2

    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning model on benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["gsm8k", "humaneval", "math", "all"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of problems to evaluate (None for all)",
    )
    parser.add_argument(
        "--use-ttc",
        action="store_true",
        help="Use test-time compute (Best-of-N sampling)",
    )
    parser.add_argument(
        "--ttc-samples",
        type=int,
        default=16,
        help="Number of samples for test-time compute",
    )
    parser.add_argument(
        "--ttc-oracle-verify",
        action="store_true",
        help="Use ground-truth verifier reranking during TTC (analysis only, not a valid benchmark setting)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero if any benchmark items error out",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Configuration file used to derive optional-track policy",
    )
    
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        track_policy = apply_track_policy(config)
    else:
        track_policy = None

    from evaluation import (
        GSM8KEvaluator,
        HumanEvalEvaluator,
        MATHEvaluator,
        run_all_benchmarks,
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_metadata = build_run_metadata(args)
    if track_policy is not None:
        run_metadata["enabled_optional_tracks"] = track_policy.enabled_optional_tracks()
    run_artifacts = RunArtifacts(
        root_output_dir=args.output_dir,
        dataset=args.benchmark,
        training_method="evaluation",
        pipeline="benchmark_evaluation",
        config=run_metadata,
        nested=False,
    )
    
    logger.info(f"Evaluating model: {args.model}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Test-time compute: {args.use_ttc}")
    
    if args.benchmark == "all":
        results = run_all_benchmarks(
            model_name=args.model,
            output_dir=args.output_dir,
            use_ttc=args.use_ttc,
            gsm8k_subset_size=args.subset_size,
            humaneval_subset_size=args.subset_size,
            ttc_samples=args.ttc_samples,
            ttc_oracle_verify=args.ttc_oracle_verify,
            run_metadata=run_metadata,
        )
        
        for name, result in results.items():
            logger.info(f"{name}: {result.accuracy:.2%} accuracy")
            logger.info(f"{name}: status={result.status} valid_run={result.valid_run}")
            output_path = f"{args.output_dir}/{name}_results.json"
            record_benchmark_result(
                run_artifacts,
                benchmark_name=name,
                result=result,
                output_path=output_path,
            )

        persist_run_manifest(args.output_dir, run_metadata)
        run_artifacts.finalize(
            summarize_suite_status(results),
            summary={
                name: result.to_dict()
                for name, result in results.items()
            },
        )
        if any(result.status == "failed" for result in results.values()):
            raise SystemExit(2)
        if args.fail_on_errors and any(result.errors > 0 for result in results.values()):
            raise SystemExit(2)
    
    elif args.benchmark == "gsm8k":
        evaluator = GSM8KEvaluator(
            model_name=args.model,
            use_test_time_compute=args.use_ttc,
            ttc_samples=args.ttc_samples,
            ttc_oracle_verify=args.ttc_oracle_verify,
        )
        evaluator.run_metadata = run_metadata
        result = evaluator.evaluate(
            split=args.split,
            subset_size=args.subset_size,
        )
        exit_code = finalize_result(
            result,
            f"{args.output_dir}/gsm8k_results.json",
            args.output_dir,
            args.fail_on_errors,
        )
        record_benchmark_result(
            run_artifacts,
            benchmark_name="gsm8k",
            result=result,
            output_path=f"{args.output_dir}/gsm8k_results.json",
        )
        run_artifacts.finalize(result.status, summary=result.to_dict())
        
        logger.info(f"GSM8K Results:")
        logger.info(f"  Accuracy: {result.accuracy:.2%}")
        logger.info(f"  Correct: {result.correct}/{result.total_problems}")
        logger.info(f"  Avg time: {result.avg_time_per_problem:.2f}s")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Valid run: {result.valid_run}")
        raise SystemExit(exit_code)
    
    elif args.benchmark == "humaneval":
        evaluator = HumanEvalEvaluator(
            model_name=args.model,
        )
        evaluator.run_metadata = run_metadata
        result = evaluator.evaluate(subset_size=args.subset_size)
        exit_code = finalize_result(
            result,
            f"{args.output_dir}/humaneval_results.json",
            args.output_dir,
            args.fail_on_errors,
        )
        record_benchmark_result(
            run_artifacts,
            benchmark_name="humaneval",
            result=result,
            output_path=f"{args.output_dir}/humaneval_results.json",
        )
        run_artifacts.finalize(result.status, summary=result.to_dict())
        
        logger.info(f"HumanEval Results:")
        logger.info(f"  Pass@1: {result.accuracy:.2%}")
        logger.info(f"  Correct: {result.correct}/{result.total_problems}")
        logger.info(f"  Status: {result.status}")
        raise SystemExit(exit_code)
    
    elif args.benchmark == "math":
        evaluator = MATHEvaluator(
            model_name=args.model,
            use_test_time_compute=args.use_ttc,
            ttc_samples=args.ttc_samples,
            ttc_oracle_verify=args.ttc_oracle_verify,
        )
        evaluator.run_metadata = run_metadata
        result = evaluator.evaluate(
            split=args.split,
            subset_size=args.subset_size,
        )
        exit_code = finalize_result(
            result,
            f"{args.output_dir}/math_results.json",
            args.output_dir,
            args.fail_on_errors,
        )
        record_benchmark_result(
            run_artifacts,
            benchmark_name="math",
            result=result,
            output_path=f"{args.output_dir}/math_results.json",
        )
        run_artifacts.finalize(result.status, summary=result.to_dict())
        
        logger.info(f"MATH Results:")
        logger.info(f"  Accuracy: {result.accuracy:.2%}")
        logger.info(f"  Correct: {result.correct}/{result.total_problems}")
        logger.info(f"  Status: {result.status}")
        raise SystemExit(exit_code)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
