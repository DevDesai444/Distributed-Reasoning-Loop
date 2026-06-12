"""Paraphrase is deterministic and preserves the gold answer."""

from random import Random

from evaluation.robustness.perturbations import Paraphrase


def test_and_then_rule_fires():
    pert = Paraphrase()
    problem = "She walked to the store and then bought milk."
    res = pert.apply(problem, "0", Random(0))
    assert res.applicable is True
    assert "and then" not in res.perturbed_problem.lower()
    assert res.new_answer is None  # label preserved


def test_no_rule_marks_inapplicable():
    pert = Paraphrase()
    # nothing in this sentence triggers any of the rules
    res = pert.apply("Five plus three is what.", "8", Random(0))
    assert res.applicable is False


def test_determinism():
    pert = Paraphrase()
    problem = "She buys 5 books and then reads them all."
    r1 = pert.apply(problem, "5", Random(7))
    r2 = pert.apply(problem, "5", Random(7))
    assert r1.perturbed_problem == r2.perturbed_problem
    assert r1.metadata.get("rule") == r2.metadata.get("rule")
