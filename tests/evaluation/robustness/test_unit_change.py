"""UnitChange scales the gold by the conversion factor."""

from random import Random

from evaluation.robustness.perturbations import UnitChange


def test_dollars_to_cents():
    pert = UnitChange()
    res = pert.apply(
        "He earned 5 dollars selling lemonade. How many dollars total?",
        "5",
        Random(0),
    )
    assert res.applicable is True
    assert "cents" in res.perturbed_problem
    assert "dollars" not in res.perturbed_problem.lower()
    assert int(res.new_answer) == 500


def test_hours_to_minutes():
    pert = UnitChange()
    res = pert.apply(
        "The trip took 2 hours each way. How long round trip in hours?",
        "4",
        Random(0),
    )
    assert res.applicable is True
    assert "minutes" in res.perturbed_problem
    assert int(res.new_answer) == 240


def test_kg_to_g():
    pert = UnitChange()
    res = pert.apply(
        "The bag holds 3 kg of rice. How many kg in two bags?",
        "6",
        Random(0),
    )
    assert res.applicable is True
    assert int(res.new_answer) == 6000


def test_skips_when_no_unit():
    pert = UnitChange()
    res = pert.apply("She had 5 apples and ate 2. How many remain?", "3",
                     Random(0))
    assert res.applicable is False
    assert res.metadata.get("reason") == "no_recognized_unit"
