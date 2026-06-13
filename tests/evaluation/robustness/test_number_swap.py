"""NumberSwap behaviour on synthetic and GSM8K-style problems."""

from random import Random

from evaluation.robustness.perturbations import NumberSwap


GSM8K_LIKE_PROBLEM = (
    "Sarah buys 4 packs of stickers. Each pack costs 3 dollars. "
    "How much does she spend?"
)
GSM8K_LIKE_SOLUTION = (
    "She spends 4 * 3 = <<4*3=12>>12 dollars.\n#### 12"
)


def test_swap_scales_answer_linearly():
    pert = NumberSwap()
    pert._current_solution = GSM8K_LIKE_SOLUTION
    rng = Random(0)
    res = pert.apply(GSM8K_LIKE_PROBLEM, "12", rng)
    assert res.applicable is True
    # the swapped operand was either "4" or "3"; the new answer should be
    # the new operand times the surviving one
    new_n = int(res.metadata["new_operand"])
    swapped = int(res.metadata["swapped_operand"])
    surviving = 12 // swapped
    assert int(res.new_answer) == new_n * surviving


def test_swap_marks_inapplicable_without_solution():
    pert = NumberSwap()
    pert._current_solution = None
    res = pert.apply(GSM8K_LIKE_PROBLEM, "12", Random(0))
    assert res.applicable is False
    assert res.metadata.get("reason") == "no_solution"


def test_swap_marks_inapplicable_for_non_multiplicative():
    pert = NumberSwap()
    pert._current_solution = "5 + 7 = <<5+7=12>>12.\n#### 12"
    res = pert.apply("She has 5 apples and gets 7 more.", "12", Random(0))
    assert res.applicable is False
    assert res.metadata.get("reason") == "non_multiplicative_final"


def test_swap_skips_operand_equal_to_gold():
    pert = NumberSwap()
    # gold is 12, and one operand is 12 itself — must skip it
    pert._current_solution = "1 * 12 = <<1*12=12>>12.\n#### 12"
    res = pert.apply("She has 1 box of 12 eggs.", "12", Random(0))
    # only "1" is a swap candidate, and it doesn't equal gold; should be applicable
    assert res.applicable is True
    assert res.metadata["swapped_operand"] == "1"
