"""Lock prompt versions down and check that they carry the fields the parser
expects."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation.judges import (
    CODE_RUBRIC_V1,
    MATH_RUBRIC_V1,
    PAIRWISE_RUBRIC_V1,
    PROMPT_VERSIONS,
    Verdict,
    get_rubric,
    parse_judge_response,
)


def test_math_rubric_has_required_fields():
    required = ["{problem}", "{response}", "{reference_answer}",
                "VERDICT:", "CONFIDENCE:", "REASON_CODE:", "RATIONALE:"]
    for token in required:
        assert token in MATH_RUBRIC_V1, f"math rubric missing {token!r}"


def test_code_rubric_has_required_fields():
    required = ["{problem}", "{response}", "{reference_answer}",
                "VERDICT:", "CONFIDENCE:", "REASON_CODE:", "RATIONALE:"]
    for token in required:
        assert token in CODE_RUBRIC_V1, f"code rubric missing {token!r}"


def test_pairwise_rubric_has_required_fields():
    for token in ("{problem}", "{response_a}", "{response_b}", "WINNER:"):
        assert token in PAIRWISE_RUBRIC_V1


def test_version_tags_stable():
    # versioning rule: the public tag must contain V1 and the rubric ref
    assert PROMPT_VERSIONS["math"][0] == "MATH_RUBRIC_V1"
    assert PROMPT_VERSIONS["code"][0] == "CODE_RUBRIC_V1"
    assert PROMPT_VERSIONS["pairwise"][0] == "PAIRWISE_RUBRIC_V1"


def test_get_rubric_unknown_type_raises():
    import pytest

    with pytest.raises(ValueError):
        get_rubric("doesnotexist")


def test_parser_extracts_fields():
    response = (
        "VERDICT: ACCEPT\n"
        "CONFIDENCE: 0.85\n"
        "REASON_CODE: NONE\n"
        "RATIONALE: looks good"
    )
    parsed = parse_judge_response(response)
    assert parsed["verdict"] == Verdict.ACCEPT
    assert parsed["confidence"] == 0.85
    assert parsed["reason_code"] == "NONE"
    assert "looks good" in parsed["rationale"]


def test_parser_handles_reject_with_reason():
    response = (
        "VERDICT: REJECT\n"
        "CONFIDENCE: 0.5\n"
        "REASON_CODE: INCOMPLETE_REASONING\n"
        "RATIONALE: skipped a step"
    )
    parsed = parse_judge_response(response)
    assert parsed["verdict"] == Verdict.REJECT
    assert parsed["reason_code"] == "INCOMPLETE_REASONING"


def test_parser_unknown_reason_becomes_other():
    response = (
        "VERDICT: REJECT\n"
        "CONFIDENCE: 0.1\n"
        "REASON_CODE: SOMETHING_INVENTED\n"
        "RATIONALE: ..."
    )
    parsed = parse_judge_response(response)
    assert parsed["reason_code"] == "OTHER"


def test_parser_malformed_defaults_to_uncertain():
    parsed = parse_judge_response("the model rambled without using the format")
    assert parsed["verdict"] == Verdict.UNCERTAIN


def test_accept_clears_reason_code():
    response = (
        "VERDICT: ACCEPT\n"
        "CONFIDENCE: 0.9\n"
        "REASON_CODE: WRONG_METHOD\n"
        "RATIONALE: shouldn't be set"
    )
    parsed = parse_judge_response(response)
    assert parsed["reason_code"] == "NONE"
