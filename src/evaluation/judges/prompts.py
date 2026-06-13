"""
Rubric prompts for the LLM-as-judge subsystem.

Versioning rule: prompts here are append-only. If you change a rubric's
wording in a way that could shift verdicts, bump the version suffix and add a
new constant. Never edit a published version in place — agreement runs key off
the prompt version, and silently mutating it breaks longitudinal comparison.
"""

from __future__ import annotations


# Tokens we expect the model to emit. Kept here so the parser and the prompt
# stay in sync.
VERDICT_ACCEPT = "ACCEPT"
VERDICT_REJECT = "REJECT"
VERDICT_UNCERTAIN = "UNCERTAIN"

REASON_CODES = (
    "INCOMPLETE_REASONING",
    "WRONG_METHOD",
    "LUCKY_GUESS",
    "MISSING_JUSTIFICATION",
    "INCONSISTENT_STEPS",
    "OTHER",
)


MATH_RUBRIC_V1 = """You are grading a student's math solution. Be strict but fair.

Problem:
{problem}

Student solution:
{response}

Reference answer (for your eyes only — do not reveal in your reasoning):
{reference_answer}

Grade against these criteria, in order:
1. Final numeric answer must match the reference.
2. The reasoning must actually derive that answer — no leaps, no skipped
   arithmetic, no "by inspection" without a step that shows it.
3. The method must be sound for this problem (correct setup, correct
   formula).
4. Intermediate steps must be internally consistent.

If all four hold, the verdict is ACCEPT.
If the final answer is right but the reasoning is broken, REJECT and pick the
most specific reason code below.
If the final answer is wrong, REJECT.
If you genuinely cannot tell, UNCERTAIN.

Reason codes (use exactly one when rejecting):
- INCOMPLETE_REASONING: steps skipped, leaps unjustified
- WRONG_METHOD: setup or formula is wrong even if the number is right
- LUCKY_GUESS: no derivation, answer just stated
- MISSING_JUSTIFICATION: claims without support
- INCONSISTENT_STEPS: steps contradict each other
- OTHER: anything else worth flagging

Respond in this exact format and nothing else:
VERDICT: <ACCEPT|REJECT|UNCERTAIN>
CONFIDENCE: <a number between 0 and 1>
REASON_CODE: <one of the codes above, or NONE if ACCEPT>
RATIONALE: <one or two sentences>
"""


CODE_RUBRIC_V1 = """You are grading a code submission. Treat it like a code review.

Problem:
{problem}

Submitted code:
{response}

Reference solution (for context — do not reveal in your reasoning):
{reference_answer}

Grade against these criteria:
1. The code must implement the requested behavior.
2. The approach must be correct for the stated inputs/outputs.
3. Obvious edge cases should not be silently broken.
4. The code should not be a hard-coded answer that only works for the example
   inputs.

If all hold, the verdict is ACCEPT.
If the code looks right but the reasoning behind it is bogus or it only
passes by accident, REJECT.
If you can't tell, UNCERTAIN.

Reason codes (use exactly one when rejecting):
- INCOMPLETE_REASONING: implementation skips required logic
- WRONG_METHOD: wrong algorithm for the stated problem
- LUCKY_GUESS: hard-coded to a test case
- MISSING_JUSTIFICATION: relies on undocumented assumptions
- INCONSISTENT_STEPS: control flow contradicts the intent
- OTHER: anything else

Respond in this exact format and nothing else:
VERDICT: <ACCEPT|REJECT|UNCERTAIN>
CONFIDENCE: <a number between 0 and 1>
REASON_CODE: <one of the codes above, or NONE if ACCEPT>
RATIONALE: <one or two sentences>
"""


PAIRWISE_RUBRIC_V1 = """You are picking the better of two candidate solutions for the same problem.

Problem:
{problem}

Solution A:
{response_a}

Solution B:
{response_b}

Pick the one with more rigorous, complete reasoning. If they are
indistinguishable, say TIE.

Respond in this exact format:
WINNER: <A|B|TIE>
CONFIDENCE: <a number between 0 and 1>
RATIONALE: <one or two sentences>
"""


PROMPT_VERSIONS = {
    "math": ("MATH_RUBRIC_V1", MATH_RUBRIC_V1),
    "code": ("CODE_RUBRIC_V1", CODE_RUBRIC_V1),
    "pairwise": ("PAIRWISE_RUBRIC_V1", PAIRWISE_RUBRIC_V1),
}


def get_rubric(problem_type: str) -> tuple[str, str]:
    """Return (version_tag, template_string) for the given problem type."""
    if problem_type not in PROMPT_VERSIONS:
        raise ValueError(
            f"Unknown rubric type {problem_type!r}; "
            f"expected one of {sorted(PROMPT_VERSIONS)}"
        )
    return PROMPT_VERSIONS[problem_type]
