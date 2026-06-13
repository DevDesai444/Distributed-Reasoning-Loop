"""DistractorSentence inserts a sentence with a non-gold number."""

from random import Random

from evaluation.robustness.perturbations import DistractorSentence


def test_inserts_in_middle_when_multiple_sentences():
    pert = DistractorSentence()
    problem = ("Tom has 4 marbles. Jerry gives him 3 more. "
               "How many marbles does Tom have now?")
    res = pert.apply(problem, "7", Random(0))
    assert res.applicable is True
    # original first sentence still leads, distractor follows
    assert res.perturbed_problem.startswith("Tom has 4 marbles.")
    assert "How many marbles" in res.perturbed_problem


def test_distractor_number_is_never_the_gold():
    pert = DistractorSentence()
    problem = "She had 11 books. She bought 1 more. Total?"
    # gold deliberately matches one of the candidate distractor numbers
    res = pert.apply(problem, "11", Random(1))
    assert res.applicable is True
    assert res.metadata["distractor_number"] != 11


def test_determinism_under_fixed_seed():
    pert = DistractorSentence()
    problem = "Tom has 4 marbles. Jerry gives him 3. Total?"
    r1 = pert.apply(problem, "7", Random(42))
    r2 = pert.apply(problem, "7", Random(42))
    assert r1.perturbed_problem == r2.perturbed_problem
    assert r1.metadata["distractor_number"] == r2.metadata["distractor_number"]
