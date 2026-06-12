"""IrrelevantContext prepends a sentence and preserves the gold."""

from random import Random

from evaluation.robustness.perturbations import (
    IrrelevantContext,
    _IRRELEVANT_SENTENCES,
)


def test_prepends_a_known_sentence():
    pert = IrrelevantContext()
    problem = "Tom has 4 marbles. How many?"
    res = pert.apply(problem, "4", Random(0))
    assert res.applicable is True
    assert res.new_answer is None
    # the original problem text survives intact at the tail
    assert res.perturbed_problem.endswith(problem)
    # the prepended sentence comes from the curated list
    injected = res.metadata["injected"]
    assert injected in _IRRELEVANT_SENTENCES
    assert res.perturbed_problem.startswith(injected)


def test_determinism_under_seed():
    pert = IrrelevantContext()
    r1 = pert.apply("x. y. z.", "0", Random(5))
    r2 = pert.apply("x. y. z.", "0", Random(5))
    assert r1.perturbed_problem == r2.perturbed_problem
