"""Reordering only swaps when no pronominal dependency exists."""

from random import Random

from evaluation.robustness.perturbations import Reordering


def test_swaps_first_two_when_safe():
    pert = Reordering()
    problem = (
        "Tom has 4 marbles. Jerry has 3 marbles. "
        "How many marbles do they have altogether?"
    )
    res = pert.apply(problem, "7", Random(0))
    assert res.applicable is True
    assert res.perturbed_problem.startswith("Jerry has 3 marbles.")
    assert res.new_answer is None


def test_skips_when_pronoun_in_first_sentence():
    pert = Reordering()
    problem = (
        "She has 4 marbles. Jerry has 3 marbles. "
        "How many marbles do they have altogether?"
    )
    res = pert.apply(problem, "7", Random(0))
    assert res.applicable is False
    assert res.metadata.get("reason") == "pronoun_in_s1"


def test_skips_when_second_sentence_starts_with_pronoun():
    pert = Reordering()
    problem = (
        "Tom has 4 marbles. He buys 3 more marbles. "
        "How many marbles does he have altogether?"
    )
    res = pert.apply(problem, "7", Random(0))
    assert res.applicable is False
    assert res.metadata.get("reason") == "pronoun_at_s2_start"


def test_skips_when_fewer_than_three_sentences():
    pert = Reordering()
    res = pert.apply("Tom has 4 marbles. He counts them.", "4", Random(0))
    assert res.applicable is False
