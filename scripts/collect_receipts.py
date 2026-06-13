#!/usr/bin/env python3
"""
Drive the synthetic / agreement / robustness pipelines and wrap each
output directory in a versioned receipt JSON.

Two modes:

  --mode smoke    Tiny N, stub generator and stub judge, CPU-only.
                  Reuses the existing smoke scripts (which already
                  emit artifact directories with the right shape) and
                  layers receipts on top.

  --mode full     Production sizes. Shells out to `main.py generate`,
                  `main.py eval agreement`, `main.py eval robust` and
                  reads the produced artifact directories.

Stages: synthetic, agreement, robustness, all, validate.
The `validate` stage re-reads any receipts already present under
<output-dir>/receipts/ and runs schema validation without launching
new runs — used by the Kaggle notebook to confirm the package is sane
before download.

The collector is intentionally crash-safe: each stage is wrapped so
that a failure in one stage doesn't wipe receipts from previously
completed stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scripts.receipt_schemas import (  # noqa: E402
    AgreementReceipt,
    RobustnessReceipt,
    SyntheticReceipt,
    receipt_path,
    load_receipt,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("collect_receipts")


STAGES = ("synthetic", "agreement", "robustness", "all", "validate")


def _now_run_id(prefix: str) -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}_%Y%m%dT%H%M%SZ")


def _hardware_dict() -> Dict[str, Any]:
    """Best-effort hardware snapshot. Receipt consumers care mostly about
    the GPU label; everything else is gravy."""
    info: Dict[str, Any] = {"gpu": None, "vram_gb": None, "cpu_only": True}
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            info["cpu_only"] = False
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        pass
    return info


# ---- smoke stage adapters -------------------------------------------------


def _run_synthetic_smoke(output_dir: Path) -> Tuple[Path, float]:
    """Run smoke_synthetic_v1.py into our own scratch dir and return
    the artifact directory + wall-clock seconds."""
    artifact_dir = output_dir / "synthetic_run"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # We import the module rather than subprocess so we get a stack trace
    # straight back on failure.
    from scripts.smoke_synthetic_v1 import main as smoke_main
    rc = smoke_main(["--output-dir", str(artifact_dir)])
    if rc != 0:
        raise RuntimeError(f"synthetic smoke returned non-zero: {rc}")
    return artifact_dir, time.time() - t0


def _run_agreement_smoke(output_dir: Path) -> Tuple[Path, float]:
    """Run the agreement-smoke pipeline into our own directory. The smoke
    script hard-codes `samples/agreement_smoke/` so we redirect by
    monkey-patching: simpler to just call its helpers directly here."""
    artifact_dir = output_dir / "agreement_run"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Re-use the smoke-script primitives but write into artifact_dir.
    from data_generator.dataset_loader import Problem  # type: ignore
    from evaluation.agreement import (
        AgreementReport,
        bin_for_problem,
        build_pair_report,
    )
    from evaluation.judges import JudgeConfig, SingleModelJudge
    from evaluation.shortcut_detector import detect_shortcuts, write_shortcuts_jsonl
    from verifier import GSM8KVerifier  # type: ignore
    from scripts.agreement_smoke import (
        SYNTH_PROBLEMS,
        SYNTH_TRACES,
        _stub_judge_fn,
    )

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

    verifier = GSM8KVerifier()
    verifier_verdicts: List[Optional[bool]] = []
    for trace in SYNTH_TRACES:
        problem = problems_by_id[trace["problem_id"]]
        r = verifier.verify_reasoning_path(trace["reasoning"], problem.answer)
        if r.status.value == "correct":
            verifier_verdicts.append(True)
        elif r.status.value == "incorrect":
            verifier_verdicts.append(False)
        else:
            verifier_verdicts.append(None)

    judge = SingleModelJudge(JudgeConfig(model_name="stub-judge", backend="stub"))
    judge.set_stub(_stub_judge_fn)

    judge_verdicts = []
    for trace in SYNTH_TRACES:
        problem = problems_by_id[trace["problem_id"]]
        judge_verdicts.append(judge.judge(
            problem=problem.problem,
            response=trace["reasoning"],
            reference_answer=problem.answer,
            problem_type="math",
        ))
    judge_bool: List[Optional[bool]] = [v.as_bool for v in judge_verdicts]

    pid_to_bucket = {p.id: bin_for_problem(p) for p in problems}
    indices_by_bin: Dict[str, List[int]] = {"low": [], "mid": [], "high": []}
    for i, trace in enumerate(SYNTH_TRACES):
        indices_by_bin[pid_to_bucket.get(trace["problem_id"], "low")].append(i)

    pair_reports = [
        build_pair_report(
            "verifier", "judge",
            verifier_verdicts, judge_bool,
            bins=indices_by_bin, n_bootstrap=200,
        ),
    ]

    report = AgreementReport(
        run_id=_now_run_id("agreement"),
        model="(synthetic)",
        benchmark="gsm8k-synthetic",
        n_problems=len(problems),
        n_traces=len(SYNTH_TRACES),
        judge_model="stub-judge",
        judge_prompt_version=judge_verdicts[0].prompt_version,
        pair_reports=pair_reports,
        metadata={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "judge_backend": "stub",
            "split": "synthetic",
            "note": "collect_receipts smoke run",
        },
    )

    (artifact_dir / "agreement_results.json").write_text(report.render_json())
    (artifact_dir / "report.md").write_text(report.render_markdown())

    records, summary = detect_shortcuts(
        problem_ids=[t["problem_id"] for t in SYNTH_TRACES],
        problems=[problems_by_id[t["problem_id"]].problem for t in SYNTH_TRACES],
        responses=[t["reasoning"] for t in SYNTH_TRACES],
        verifier_verdicts=verifier_verdicts,
        judge_verdicts=judge_verdicts,
        difficulty_bins=pid_to_bucket,
    )
    write_shortcuts_jsonl(records, artifact_dir / "shortcuts.jsonl")
    (artifact_dir / "shortcuts_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2)
    )

    return artifact_dir, time.time() - t0


def _run_robustness_smoke(output_dir: Path) -> Tuple[Path, float]:
    artifact_dir = output_dir / "robustness_run"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from evaluation.robustness import default_perturbations, evaluate_robustness  # noqa: E402
    from verifier import GSM8KVerifier, VerificationStatus  # type: ignore  # noqa: E402
    from scripts.robustness_smoke import SYNTH_PROBLEMS, _make_stub_generate

    verifier = GSM8KVerifier()

    def verify_fn(out: str, gold: str) -> Optional[bool]:
        try:
            r = verifier.verify_reasoning_path(out, gold)
            if r.status == VerificationStatus.CORRECT:
                return True
            if r.status == VerificationStatus.INCORRECT:
                return False
            return None
        except Exception:
            return None

    generate_fn = _make_stub_generate(SYNTH_PROBLEMS)
    report = evaluate_robustness(
        problems=SYNTH_PROBLEMS,
        n_samples=1,
        perturbations=default_perturbations(),
        generate_fn=generate_fn,
        verify_fn=verify_fn,
        seed=0,
        n_bootstrap=100,
        model_label="smoke-stub",
        benchmark_label="gsm8k-synthetic",
        run_id=_now_run_id("robustness"),
        extra_metadata={
            "split": "synthetic",
            "note": "collect_receipts smoke run",
        },
    )

    (artifact_dir / "robustness_results.json").write_text(report.render_json())
    (artifact_dir / "report.md").write_text(report.render_markdown())
    manifest = {
        "run_id": report.run_id,
        "kind": "robustness",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "smoke-stub",
        "benchmark": "gsm8k-synthetic",
        "split": "synthetic",
        "n_problems": len(SYNTH_PROBLEMS),
        "n_samples_per_problem": 1,
        "seed": 0,
        "overall_robustness": report.overall_robustness,
    }
    (artifact_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    return artifact_dir, time.time() - t0


# ---- full-stage shellouts -------------------------------------------------


def _run_main(argv: List[str]) -> None:
    """Shell out to `python main.py ...` and raise on non-zero."""
    cmd = [sys.executable, str(_REPO_ROOT / "main.py")] + argv
    logger.info("running: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"command failed (rc={res.returncode}): {' '.join(cmd)}")


def _run_synthetic_full(
    output_dir: Path,
    *,
    generator_model: str,
    target_pairs: int,
    num_paths: int,
    max_pairs_per_problem: int,
) -> Tuple[Path, float]:
    artifact_dir = output_dir / "synthetic_run"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _run_main([
        "generate",
        "--dataset", "gsm8k",
        "--model", generator_model,
        "--num-paths", str(num_paths),
        "--target-pairs", str(target_pairs),
        "--max-pairs-per-problem", str(max_pairs_per_problem),
        "--output-dir", str(artifact_dir),
        "--with-provenance",
    ])
    return artifact_dir, time.time() - t0


def _run_agreement_full(
    output_dir: Path,
    *,
    generator_model: str,
    judge_model: str,
    n_problems: int,
    n_samples: int,
) -> Tuple[Path, float]:
    artifact_dir = output_dir / "agreement_run"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _run_main([
        "eval", "agreement",
        "--model", generator_model,
        "--judge-model", judge_model,
        "--benchmark", "gsm8k",
        "--n-problems", str(n_problems),
        "--n-samples", str(n_samples),
        "--output-dir", str(artifact_dir),
    ])
    return artifact_dir, time.time() - t0


def _run_robustness_full(
    output_dir: Path,
    *,
    generator_model: str,
    n_problems: int,
    n_samples: int,
    seed: int,
) -> Tuple[Path, float]:
    artifact_dir = output_dir / "robustness_run"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _run_main([
        "eval", "robust",
        "--model", generator_model,
        "--benchmark", "gsm8k",
        "--n-problems", str(n_problems),
        "--n-samples", str(n_samples),
        "--seed", str(seed),
        "--output-dir", str(artifact_dir),
    ])
    return artifact_dir, time.time() - t0


# ---- receipt construction -------------------------------------------------


def _write_synthetic_receipt(
    artifact_dir: Path,
    receipts_dir: Path,
    *,
    wall_clock_seconds: float,
    hardware: Dict[str, Any],
    model: str,
    target_pairs: Optional[int],
    run_id: Optional[str],
    notes: Dict[str, Any],
) -> Path:
    receipt = SyntheticReceipt.from_run_artifacts(
        artifact_dir,
        wall_clock_seconds=wall_clock_seconds,
        hardware=hardware,
        model=model,
        target_pairs=target_pairs,
        run_id=run_id,
        notes=notes,
    )
    receipt.validate()
    out = receipts_dir / "synthetic_v1.json"
    receipt.to_json(out)
    return out


def _write_agreement_receipt(
    artifact_dir: Path,
    receipts_dir: Path,
    *,
    wall_clock_seconds: float,
    hardware: Dict[str, Any],
    notes: Dict[str, Any],
) -> Path:
    receipt = AgreementReceipt.from_run_artifacts(
        artifact_dir,
        wall_clock_seconds=wall_clock_seconds,
        hardware=hardware,
        notes=notes,
    )
    receipt.validate()
    out = receipts_dir / "agreement_v1.json"
    receipt.to_json(out)
    return out


def _write_robustness_receipt(
    artifact_dir: Path,
    receipts_dir: Path,
    *,
    wall_clock_seconds: float,
    hardware: Dict[str, Any],
    notes: Dict[str, Any],
) -> Path:
    receipt = RobustnessReceipt.from_run_artifacts(
        artifact_dir,
        wall_clock_seconds=wall_clock_seconds,
        hardware=hardware,
        notes=notes,
    )
    receipt.validate()
    out = receipts_dir / "robustness_v1.json"
    receipt.to_json(out)
    return out


# ---- dispatch -------------------------------------------------------------


def _stage_synthetic(args, output_dir: Path, receipts_dir: Path) -> Path:
    hardware = _hardware_dict()
    if args.mode == "smoke":
        artifact_dir, wall = _run_synthetic_smoke(output_dir)
        target_pairs = None
        model = "stub-canned-traces"
        notes = {"mode": "smoke"}
    else:
        artifact_dir, wall = _run_synthetic_full(
            output_dir,
            generator_model=args.generator_model,
            target_pairs=args.target_pairs,
            num_paths=args.num_paths,
            max_pairs_per_problem=args.max_pairs_per_problem,
        )
        target_pairs = args.target_pairs
        model = args.generator_model
        notes = {"mode": "full"}
    return _write_synthetic_receipt(
        artifact_dir, receipts_dir,
        wall_clock_seconds=wall,
        hardware=hardware,
        model=model,
        target_pairs=target_pairs,
        run_id=_now_run_id("synthetic"),
        notes=notes,
    )


def _stage_agreement(args, output_dir: Path, receipts_dir: Path) -> Path:
    hardware = _hardware_dict()
    if args.mode == "smoke":
        artifact_dir, wall = _run_agreement_smoke(output_dir)
        notes = {"mode": "smoke", "judge_backend": "stub"}
    else:
        artifact_dir, wall = _run_agreement_full(
            output_dir,
            generator_model=args.generator_model,
            judge_model=args.judge_model,
            n_problems=args.n_problems,
            n_samples=args.n_samples,
        )
        notes = {"mode": "full"}
    return _write_agreement_receipt(
        artifact_dir, receipts_dir,
        wall_clock_seconds=wall,
        hardware=hardware,
        notes=notes,
    )


def _stage_robustness(args, output_dir: Path, receipts_dir: Path) -> Path:
    hardware = _hardware_dict()
    if args.mode == "smoke":
        artifact_dir, wall = _run_robustness_smoke(output_dir)
        notes = {"mode": "smoke"}
    else:
        artifact_dir, wall = _run_robustness_full(
            output_dir,
            generator_model=args.generator_model,
            n_problems=args.n_problems,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        notes = {"mode": "full"}
    return _write_robustness_receipt(
        artifact_dir, receipts_dir,
        wall_clock_seconds=wall,
        hardware=hardware,
        notes=notes,
    )


def _stage_validate(receipts_dir: Path) -> List[Path]:
    """Re-read every *.json under receipts/ and run validate(). Used by
    the Kaggle notebook as a final sanity gate before download."""
    found: List[Path] = []
    if not receipts_dir.exists():
        raise FileNotFoundError(f"no receipts directory at {receipts_dir}")
    for path in sorted(receipts_dir.glob("*.json")):
        receipt = load_receipt(path)
        receipt.validate()
        logger.info("validated %s (%s)", path.name, receipt.kind)
        found.append(path)
    if not found:
        raise RuntimeError(f"no receipt files found under {receipts_dir}")
    return found


# ---- argparse / main ------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Top-level scratch directory for this run.")
    p.add_argument("--generator-model", type=str,
                   default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Policy / generator model for full-mode runs.")
    p.add_argument("--judge-model", type=str,
                   default="Qwen/Qwen2.5-7B-Instruct",
                   help="Judge model for full-mode agreement runs.")
    p.add_argument("--target-pairs", type=int, default=30000,
                   help="Full-mode synthetic target.")
    p.add_argument("--num-paths", type=int, default=12,
                   help="Generations per problem (synthetic full mode).")
    p.add_argument("--max-pairs-per-problem", type=int, default=8)
    p.add_argument("--n-problems", type=int, default=200,
                   help="Problems per eval (agreement / robustness).")
    p.add_argument("--n-samples", type=int, default=4,
                   help="Samples per problem (agreement / robustness).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fail-stage", type=str, default=None,
                   choices=("synthetic", "agreement", "robustness"),
                   help=argparse.SUPPRESS)
    return p


def _maybe_inject_failure(args, stage: str) -> None:
    if args.fail_stage == stage:
        raise RuntimeError(f"forced failure on stage {stage!r} (test hook)")


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts_dir = output_dir / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "validate":
        paths = _stage_validate(receipts_dir)
        for p in paths:
            print(p)
        return 0

    stages_to_run: List[str] = []
    if args.stage == "all":
        stages_to_run = ["synthetic", "agreement", "robustness"]
    else:
        stages_to_run = [args.stage]

    errors: Dict[str, str] = {}
    written: Dict[str, Path] = {}
    for stage in stages_to_run:
        logger.info("=== stage: %s (mode=%s) ===", stage, args.mode)
        try:
            _maybe_inject_failure(args, stage)
            if stage == "synthetic":
                path = _stage_synthetic(args, output_dir, receipts_dir)
            elif stage == "agreement":
                path = _stage_agreement(args, output_dir, receipts_dir)
            elif stage == "robustness":
                path = _stage_robustness(args, output_dir, receipts_dir)
            else:
                raise AssertionError(f"unreachable stage: {stage!r}")
            written[stage] = path
            logger.info("wrote %s", path)
        except Exception as exc:
            # Crash-safe: log the failure and keep going. Previously-
            # written receipts stay on disk.
            logger.exception("stage %s failed: %s", stage, exc)
            errors[stage] = str(exc)

    summary = {
        "mode": args.mode,
        "stages_requested": stages_to_run,
        "receipts_written": {k: str(v) for k, v in written.items()},
        "errors": errors,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "collector_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
