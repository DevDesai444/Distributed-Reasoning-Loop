"""
Retention runner: take a benchmark slice, run baseline + perturbed
generation, verify, and report per-perturbation pass@1 retention with
bootstrap CIs.

The runner is split so the metric-only path (verdicts in, stats out) is
testable without a model: synthetic verdicts go through `verdicts_to_stats`.
Real runs assemble verdicts via generation + verifier.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .perturbations import Perturbation, PerturbationResult, default_perturbations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PerturbationOutcome:
    """Per-problem record for a single perturbation."""

    problem_id: str
    applicable: bool
    baseline_correct: Optional[bool]      # None if baseline wasn't verifiable
    perturbed_correct: Optional[bool]     # None when not applicable
    baseline_answer: str
    perturbed_answer: Optional[str]       # may have been rewritten
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationStats:
    """Aggregate stats for one perturbation across all problems."""

    name: str
    rewrites_label: bool
    n_problems_total: int
    n_applicable: int
    applicability_rate: float
    matched_baseline_pass_at_1: float
    perturbed_pass_at_1: float
    retention: float
    retention_ci: Tuple[float, float]
    bootstrap_n: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rewrites_label": self.rewrites_label,
            "n_problems_total": self.n_problems_total,
            "n_applicable": self.n_applicable,
            "applicability_rate": self.applicability_rate,
            "matched_baseline_pass_at_1": self.matched_baseline_pass_at_1,
            "perturbed_pass_at_1": self.perturbed_pass_at_1,
            "retention": self.retention,
            "retention_ci": list(self.retention_ci),
            "bootstrap_n": self.bootstrap_n,
        }


@dataclass
class RobustnessReport:
    """End-to-end report — JSON-serializable."""

    run_id: str
    model: str
    benchmark: str
    n_problems: int
    n_samples_per_problem: int
    seed: int
    per_perturbation: List[PerturbationStats]
    overall_robustness: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "benchmark": self.benchmark,
            "n_problems": self.n_problems,
            "n_samples_per_problem": self.n_samples_per_problem,
            "seed": self.seed,
            "per_perturbation": [p.to_dict() for p in self.per_perturbation],
            "overall_robustness": self.overall_robustness,
            "metadata": self.metadata,
        }

    def render_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def render_markdown(self) -> str:
        lines = [
            f"# Robustness report — {self.benchmark}",
            "",
            f"- run id: `{self.run_id}`",
            f"- model: `{self.model}`",
            f"- problems: {self.n_problems}, samples per problem: {self.n_samples_per_problem}",
            f"- seed: {self.seed}",
            f"- overall robustness (geometric mean of retentions): "
            f"**{self.overall_robustness:.3f}**",
            "",
            "| perturbation | applicable | applic.% | baseline p@1 | perturbed p@1 | retention | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for p in self.per_perturbation:
            ci_lo, ci_hi = p.retention_ci
            lines.append(
                f"| {p.name} | {p.n_applicable}/{p.n_problems_total} "
                f"| {p.applicability_rate * 100:.1f}% "
                f"| {p.matched_baseline_pass_at_1:.3f} "
                f"| {p.perturbed_pass_at_1:.3f} "
                f"| {p.retention:.3f} "
                f"| [{ci_lo:.3f}, {ci_hi:.3f}] |"
            )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Verdict-only aggregation (testable without a model)
# ---------------------------------------------------------------------------


def _pass_at_1(verdicts: Sequence[Optional[bool]]) -> float:
    """Mean of truthy verdicts, ignoring Nones; 0.0 on empty."""
    valid = [bool(v) for v in verdicts if v is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def _bootstrap_retention_ci(
    paired: Sequence[Tuple[bool, bool]],
    *,
    n_resamples: int,
    ci: float,
    seed: int,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for the ratio perturbed_p@1 / baseline_p@1.

    `paired` is a list of (baseline_correct, perturbed_correct) tuples
    restricted to the applicable subset.
    """
    if not paired or n_resamples <= 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(paired)
    samples: List[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        base = sum(1 for i in idx if paired[i][0]) / n
        pert = sum(1 for i in idx if paired[i][1]) / n
        if base == 0:
            continue
        samples.append(pert / base)
    if not samples:
        return (0.0, 0.0)
    samples.sort()
    alpha = (1 - ci) / 2
    lo_idx = max(0, int(math.floor(alpha * len(samples))))
    hi_idx = min(len(samples) - 1, int(math.ceil((1 - alpha) * len(samples))) - 1)
    return (samples[lo_idx], samples[hi_idx])


def verdicts_to_stats(
    name: str,
    rewrites_label: bool,
    outcomes: Sequence[PerturbationOutcome],
    *,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> PerturbationStats:
    """
    Reduce a list of per-problem outcomes to a `PerturbationStats`.
    Pure function over (verdicts, applicability) — no model needed.
    """
    n_total = len(outcomes)
    applicable_outcomes = [o for o in outcomes if o.applicable]
    n_applicable = len(applicable_outcomes)
    appl_rate = n_applicable / n_total if n_total else 0.0

    # only count outcomes where both baseline and perturbed have a verdict
    paired: List[Tuple[bool, bool]] = []
    for o in applicable_outcomes:
        if o.baseline_correct is None or o.perturbed_correct is None:
            continue
        paired.append((bool(o.baseline_correct), bool(o.perturbed_correct)))

    if paired:
        baseline_p1 = sum(1 for b, _ in paired if b) / len(paired)
        perturbed_p1 = sum(1 for _, p in paired if p) / len(paired)
    else:
        baseline_p1 = 0.0
        perturbed_p1 = 0.0

    if baseline_p1 > 0:
        retention = perturbed_p1 / baseline_p1
    else:
        # No baseline correct in the applicable set — retention is undefined.
        # We report 0.0 and let the caller see baseline_p1=0 to disambiguate.
        retention = 0.0

    ci_lo, ci_hi = _bootstrap_retention_ci(
        paired, n_resamples=n_bootstrap, ci=ci, seed=seed,
    )

    return PerturbationStats(
        name=name,
        rewrites_label=rewrites_label,
        n_problems_total=n_total,
        n_applicable=n_applicable,
        applicability_rate=appl_rate,
        matched_baseline_pass_at_1=baseline_p1,
        perturbed_pass_at_1=perturbed_p1,
        retention=retention,
        retention_ci=(ci_lo, ci_hi),
        bootstrap_n=n_bootstrap,
    )


def _geometric_mean(values: Sequence[float]) -> float:
    """Geometric mean over positive values; returns 0.0 if any value is 0."""
    if not values:
        return 0.0
    log_sum = 0.0
    for v in values:
        if v <= 0:
            return 0.0
        log_sum += math.log(v)
    return math.exp(log_sum / len(values))


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------

# Type aliases for the generation / verification callbacks. We accept them
# as parameters so the runner is testable with stubs.
GenerateFn = Callable[[str, int], List[str]]
"""(problem_text, n_samples) -> list of model outputs (length n_samples)"""

VerifyFn = Callable[[str, str], Optional[bool]]
"""(model_output, gold_answer) -> True/False/None (None == unverifiable)"""


def _aggregate_sample_verdicts(verdicts: Sequence[Optional[bool]]) -> Optional[bool]:
    """
    Aggregate per-sample verdicts into a single pass@1 verdict.

    pass@1 over n samples = "did the model get it right at least once"? No —
    pass@1 in the literature is the per-sample mean. We compute per-sample
    pass and then aggregate to a problem-level boolean by majority vote
    (ties break to False — better to call a noisy result wrong).
    """
    valid = [v for v in verdicts if v is not None]
    if not valid:
        return None
    n_true = sum(1 for v in valid if v)
    if n_true * 2 > len(valid):
        return True
    return False


def evaluate_robustness(
    *,
    problems: Sequence[Any],
    n_samples: int,
    perturbations: Sequence[Perturbation],
    generate_fn: GenerateFn,
    verify_fn: VerifyFn,
    seed: int = 0,
    n_bootstrap: int = 500,
    model_label: str = "model",
    benchmark_label: str = "gsm8k",
    run_id: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> RobustnessReport:
    """
    Run the full robustness eval.

    `problems` is a sequence of `Problem`-shaped objects (must have `.id`,
    `.problem`, `.answer`, and a `.metadata` dict — see
    `data_generator.dataset_loader.Problem`).

    The runner is generation-backend-agnostic: pass any callable for
    `generate_fn`. The CLI wires up vLLM / transformers in
    `scripts/robustness_eval.py`.
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("robustness_%Y%m%dT%H%M%SZ")

    rng_master = random.Random(seed)

    per_pert_outcomes: Dict[str, List[PerturbationOutcome]] = {
        p.name: [] for p in perturbations
    }

    # Baseline generation per problem — share across perturbations.
    baseline_verdicts_by_pid: Dict[str, Optional[bool]] = {}
    for problem in problems:
        outputs = generate_fn(problem.problem, n_samples)
        sample_verdicts = [verify_fn(o, problem.answer) for o in outputs]
        baseline_verdicts_by_pid[problem.id] = _aggregate_sample_verdicts(sample_verdicts)

    # Per-perturbation: apply, generate, verify.
    for pert in perturbations:
        for problem in problems:
            # seed per (pid, pert) so reruns are stable but distinct
            pert_seed = rng_master.randrange(2**31)
            rng = random.Random(pert_seed)

            # Thread the solution into NumberSwap if it needs it.
            solution = (problem.metadata or {}).get("full_solution")
            if hasattr(pert, "_current_solution"):
                setattr(pert, "_current_solution", solution)
            elif pert.__class__.__name__ == "NumberSwap":
                setattr(pert, "_current_solution", solution)

            result: PerturbationResult = pert.apply(problem.problem, problem.answer, rng)

            if not result.applicable:
                per_pert_outcomes[pert.name].append(PerturbationOutcome(
                    problem_id=problem.id,
                    applicable=False,
                    baseline_correct=None,
                    perturbed_correct=None,
                    baseline_answer=problem.answer,
                    perturbed_answer=None,
                    metadata=dict(result.metadata),
                ))
                continue

            effective_answer = result.new_answer if result.new_answer is not None else problem.answer
            outputs = generate_fn(result.perturbed_problem, n_samples)
            sample_verdicts = [verify_fn(o, effective_answer) for o in outputs]
            pert_verdict = _aggregate_sample_verdicts(sample_verdicts)

            per_pert_outcomes[pert.name].append(PerturbationOutcome(
                problem_id=problem.id,
                applicable=True,
                baseline_correct=baseline_verdicts_by_pid[problem.id],
                perturbed_correct=pert_verdict,
                baseline_answer=problem.answer,
                perturbed_answer=effective_answer,
                metadata=dict(result.metadata),
            ))

    stats_list: List[PerturbationStats] = []
    for pert in perturbations:
        s = verdicts_to_stats(
            name=pert.name,
            rewrites_label=pert.rewrites_label,
            outcomes=per_pert_outcomes[pert.name],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        stats_list.append(s)

    overall = _geometric_mean([s.retention for s in stats_list])

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_perturbations": len(perturbations),
        "perturbation_names": [p.name for p in perturbations],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return RobustnessReport(
        run_id=run_id,
        model=model_label,
        benchmark=benchmark_label,
        n_problems=len(problems),
        n_samples_per_problem=n_samples,
        seed=seed,
        per_perturbation=stats_list,
        overall_robustness=overall,
        metadata=metadata,
    )


def per_problem_outcomes_to_jsonl(
    outcomes_by_pert: Dict[str, List[PerturbationOutcome]],
    path: Path,
) -> None:
    """Dump per-perturbation per-problem outcomes for inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for pert_name, outcomes in outcomes_by_pert.items():
            for o in outcomes:
                row = asdict(o)
                row["perturbation"] = pert_name
                fh.write(json.dumps(row) + "\n")
