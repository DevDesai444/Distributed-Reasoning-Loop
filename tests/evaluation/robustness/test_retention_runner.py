"""Retention stats and the end-to-end runner with synthetic verdicts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation.robustness.perturbations import IrrelevantContext, default_perturbations
from evaluation.robustness.runner import (
    PerturbationOutcome,
    evaluate_robustness,
    verdicts_to_stats,
)


def _outcome(pid: str, applicable: bool,
             base: Optional[bool], pert: Optional[bool]) -> PerturbationOutcome:
    return PerturbationOutcome(
        problem_id=pid,
        applicable=applicable,
        baseline_correct=base,
        perturbed_correct=pert,
        baseline_answer="42",
        perturbed_answer="42" if applicable else None,
    )


def test_retention_perfect_when_all_match():
    outs = [_outcome(f"p{i}", True, True, True) for i in range(10)]
    s = verdicts_to_stats("x", False, outs, n_bootstrap=50, seed=0)
    assert s.matched_baseline_pass_at_1 == 1.0
    assert s.perturbed_pass_at_1 == 1.0
    assert s.retention == 1.0
    assert s.applicability_rate == 1.0


def test_retention_half_when_perturbed_halves():
    outs = []
    for i in range(10):
        # baseline correct for all; perturbed correct for first 5
        outs.append(_outcome(f"p{i}", True, True, i < 5))
    s = verdicts_to_stats("x", False, outs, n_bootstrap=200, seed=0)
    assert s.matched_baseline_pass_at_1 == 1.0
    assert s.perturbed_pass_at_1 == 0.5
    assert s.retention == 0.5
    lo, hi = s.retention_ci
    assert lo <= 0.5 <= hi


def test_applicability_rate_reported():
    outs = (
        [_outcome(f"a{i}", True, True, True) for i in range(4)]
        + [_outcome(f"b{i}", False, None, None) for i in range(6)]
    )
    s = verdicts_to_stats("x", False, outs, n_bootstrap=50, seed=0)
    assert s.n_problems_total == 10
    assert s.n_applicable == 4
    assert abs(s.applicability_rate - 0.4) < 1e-9


@dataclass
class _StubProblem:
    id: str
    problem: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def test_end_to_end_with_stub_generation():
    problems = [
        _StubProblem(
            id=f"q{i}",
            problem="Tom has 4 marbles. Jerry has 3 marbles. How many marbles altogether?",
            answer="7",
        )
        for i in range(5)
    ]

    # generate "the answer is 7" always — baseline + perturbed both pass
    def generate(_text: str, n: int) -> List[str]:
        return ["The answer is 7"] * n

    # cheap verifier: True iff "7" appears
    def verify(output: str, gold: str) -> Optional[bool]:
        return gold in output

    # use only the label-preserving perturbations so we don't need solutions
    perts = [p for p in default_perturbations() if not p.rewrites_label]
    assert any(isinstance(p, IrrelevantContext) for p in perts)

    report = evaluate_robustness(
        problems=problems,
        n_samples=2,
        perturbations=perts,
        generate_fn=generate,
        verify_fn=verify,
        seed=0,
        n_bootstrap=100,
    )

    # IrrelevantContext is always applicable; retention should be 1.0
    irrelevant = next(p for p in report.per_perturbation
                      if p.name == "irrelevant_context")
    assert irrelevant.n_applicable == 5
    assert irrelevant.retention == 1.0
    assert 0.0 <= report.overall_robustness <= 1.0
