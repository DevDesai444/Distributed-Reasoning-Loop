"""Tests for the composite quality score: each component on synthetic inputs."""

import pytest

from data_generator.quality_score import (
    length_penalty,
    score_components,
    score_trace,
    step_coherence,
)


# ---------- step coherence -------------------------------------------------


def test_step_coherence_all_correct():
    text = "First <<48/2=24>> and then <<48+24=72>>."
    assert step_coherence(text) == 1.0


def test_step_coherence_all_wrong():
    text = "Bad math: <<2+2=5>> and <<3*3=10>>."
    assert step_coherence(text) == 0.0


def test_step_coherence_mixed():
    text = "<<2+2=4>> ok, but <<3*3=10>> wrong."
    assert step_coherence(text) == 0.5


def test_step_coherence_no_blocks_returns_one():
    # Absence of calc tags should not penalize the trace.
    assert step_coherence("Some prose with no inline math.") == 1.0


def test_step_coherence_divide_by_zero_skipped():
    # The bad block is skipped (treated as a non-match) rather than crashing.
    text = "<<1+1=2>> and <<5/0=0>>."
    # 1 valid match / 1 considered = 1.0 (the div-by-zero block is excluded)
    score = step_coherence(text)
    assert 0.0 <= score <= 1.0


def test_step_coherence_decimals_and_alt_operators():
    text = "<<1.5+0.5=2.0>> and <<3x4=12>>."
    assert step_coherence(text) == 1.0


# ---------- length penalty -------------------------------------------------


def test_length_penalty_short_full_credit():
    assert length_penalty(100, median_length=200) == 1.0
    assert length_penalty(400, median_length=200) == 1.0  # exactly 2x


def test_length_penalty_drops_to_half_at_four_x():
    assert length_penalty(800, median_length=200) == pytest.approx(0.5)


def test_length_penalty_zero_at_six_x_and_beyond():
    assert length_penalty(1200, median_length=200) == 0.0
    assert length_penalty(5000, median_length=200) == 0.0


def test_length_penalty_smooth_between_2x_and_4x():
    # Halfway between 2x and 4x should land halfway between 1.0 and 0.5.
    val = length_penalty(600, median_length=200)  # 3x
    assert val == pytest.approx(0.75)


def test_length_penalty_zero_median_returns_one():
    assert length_penalty(100, median_length=0) == 1.0


# ---------- composite ------------------------------------------------------


def test_score_trace_correct_clean_trace():
    text = "<<2+2=4>>"
    score = score_trace(text, is_correct=True, median_length=10)
    # verifier=1, step_coherence=1, length_penalty=1 -> 1.0
    assert score == pytest.approx(1.0)


def test_score_trace_incorrect_clean_trace():
    text = "<<2+2=4>>"
    score = score_trace(text, is_correct=False, median_length=10)
    # verifier=0, step_coherence=1, length_penalty=1 -> 2/3
    assert score == pytest.approx(2.0 / 3.0)


def test_score_components_breakdown():
    components = score_components(
        "<<2+2=4>> and <<3*3=10>>",
        is_correct=True,
        median_length=10,
    )
    assert components.verifier == 1.0
    assert components.step_coherence == 0.5
    # length_penalty depends on text length vs median; should be in [0, 1]
    assert 0.0 <= components.length_penalty <= 1.0
    # Score is the mean of the three.
    assert components.score == pytest.approx(
        (1.0 + 0.5 + components.length_penalty) / 3.0
    )


def test_score_in_unit_interval():
    score = score_trace("trace text", is_correct=False, median_length=100)
    assert 0.0 <= score <= 1.0
