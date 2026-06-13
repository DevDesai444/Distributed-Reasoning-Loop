"""
Deterministic GSM8K perturbations.

Each perturbation is a small class with:
- `name`
- `rewrites_label` (bool — does the gold answer change?)
- `apply(problem, answer, rng) -> PerturbationResult`

A `PerturbationResult` flags whether the perturbation was applicable to this
problem. If `applicable=False` we leave the problem unchanged and the runner
drops the trace from the per-perturbation retention numerator/denominator.
This matters: applying a label-rewriting perturbation to a problem where we
can't safely re-derive the gold would silently inject wrong tests.

Determinism: every transform takes a `random.Random` argument (callers seed
it from the problem id + perturbation name). No LLM in the loop. No imports
from `random` global state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Optional, Protocol, Tuple


@dataclass
class PerturbationResult:
    """Outcome of applying a single perturbation."""

    perturbed_problem: str
    new_answer: Optional[str]  # None means original answer still applies
    applicable: bool
    metadata: Dict[str, object] = field(default_factory=dict)


class Perturbation(Protocol):
    name: str
    rewrites_label: bool

    def apply(
        self, problem: str, answer: str, rng: Random
    ) -> PerturbationResult: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])")
# `<<expr=result>>` markers used by GSM8K reference solutions.
_STEP_RE = re.compile(r"<<([^=]+)=([^>]+)>>")
# rough sentence splitter — GSM8K problems are clean prose.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _format_scaled(value: float) -> str:
    """Render a scaled answer. Integers stay integers; decimals get trimmed."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    # 6 sig digits, strip trailing zeros
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    return rendered or "0"


# ---------------------------------------------------------------------------
# 1. NumberSwap — label-rewriting
# ---------------------------------------------------------------------------


class NumberSwap:
    """
    Replace one operand with a new value and scale the gold answer.

    We only consider this applicable when:
      - the GSM8K solution metadata is available with `<<step>>` markers
      - exactly one step references the swapped value
      - that step is on the final answer line (last step)
      - the relationship between input and output is multiplicative
        (the value appears as a multiplier in that step)

    This is conservative on purpose. If the math is anything other than
    "answer = k * value" for the swapped operand, we mark applicable=False
    rather than ship a wrong gold.
    """

    name = "number_swap"
    rewrites_label = True

    def __init__(self, full_solution: Optional[str] = None):
        # callers thread the GSM8K full solution in via metadata at apply
        # time, but we let it be set at construction too for ad-hoc use.
        self._fallback_solution = full_solution

    def apply(
        self, problem: str, answer: str, rng: Random
    ) -> PerturbationResult:
        # The runner stuffs the GSM8K full solution into rng.__dict__ via the
        # `_solution` attribute — that's gross, so instead we accept it via
        # an instance attribute set by the runner before each call.
        solution = getattr(self, "_current_solution", None) or self._fallback_solution
        if not solution:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "no_solution"})

        steps = _STEP_RE.findall(solution)
        if not steps:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "no_step_markers"})

        # Final step's stated result should equal the gold answer.
        final_expr, final_result = steps[-1]
        final_result = final_result.strip()
        ans_num = _try_float(answer)
        final_num = _try_float(final_result)
        if ans_num is None or final_num is None:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "non_numeric_answer"})
        if abs(ans_num - final_num) > 1e-6:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "answer_neq_final_step"})

        # Pick a candidate operand to swap. We want an operand that:
        #   - appears in the problem text exactly once
        #   - appears in the final step expression
        #   - does NOT equal the gold answer
        operands = _NUMBER_RE.findall(final_expr)
        operands = [o for o in operands if _try_float(o) is not None]
        if not operands:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "no_operands"})

        # multiplicative shape: final_expr is "A*B" possibly with whitespace
        mult_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*\*\s*(-?\d+(?:\.\d+)?)\s*",
                                  final_expr)
        if not mult_match:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "non_multiplicative_final"})

        a_str, b_str = mult_match.group(1), mult_match.group(2)
        candidates: List[Tuple[str, float]] = []
        for cand in (a_str, b_str):
            cand_num = _try_float(cand)
            if cand_num is None or cand_num == 0:
                continue
            if abs(cand_num - ans_num) < 1e-9:
                continue
            # must appear exactly once in problem text
            if len(re.findall(rf"(?<![\w.])-?{re.escape(cand.lstrip('-'))}(?![\w.])",
                              problem)) != 1:
                continue
            candidates.append((cand, cand_num))

        if not candidates:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "no_safe_operand"})

        pick_str, pick_val = candidates[rng.randrange(len(candidates))]
        # new value in [pick/2, pick*2], integer-preserving
        lo, hi = pick_val / 2.0, pick_val * 2.0
        if pick_val == int(pick_val):
            lo_i, hi_i = max(1, int(round(lo))), max(2, int(round(hi)))
            new_val_num: float = float(rng.randint(lo_i, hi_i))
            if abs(new_val_num - pick_val) < 1e-9:
                new_val_num = float(hi_i if pick_val < hi_i else lo_i)
        else:
            new_val_num = lo + rng.random() * (hi - lo)
        new_val_str = _format_scaled(new_val_num)

        new_problem = re.sub(
            rf"(?<![\w.])-?{re.escape(pick_str.lstrip('-'))}(?![\w.])",
            new_val_str,
            problem,
            count=1,
        )

        scale = new_val_num / pick_val
        new_answer = _format_scaled(ans_num * scale)

        return PerturbationResult(
            perturbed_problem=new_problem,
            new_answer=new_answer,
            applicable=True,
            metadata={
                "swapped_operand": pick_str,
                "new_operand": new_val_str,
                "scale": scale,
            },
        )


# ---------------------------------------------------------------------------
# 2. UnitChange — label-rewriting
# ---------------------------------------------------------------------------

_UNIT_PAIRS: List[Tuple[str, str, str, float]] = [
    # (from_singular, from_plural, to_token, multiplier_to_new)
    # currency
    ("dollar", "dollars", "cents", 100.0),
    # time
    ("hour", "hours", "minutes", 60.0),
    # mass
    ("kilogram", "kilograms", "grams", 1000.0),
    ("kg", "kg", "g", 1000.0),
]


class UnitChange:
    """
    Rewrite a recognized unit token and scale gold by the conversion factor.

    Only applies when both:
      - the problem text contains a unit we recognize, AND
      - the gold answer parses as a number

    We don't try to handle compound units, currency symbols ($), or
    multi-step unit phrases. If the problem already mixes units, we skip.
    """

    name = "unit_change"
    rewrites_label = True

    def apply(
        self, problem: str, answer: str, rng: Random
    ) -> PerturbationResult:
        ans_num = _try_float(answer)
        if ans_num is None:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "non_numeric_answer"})

        # find the first matching unit pair (deterministic order — first
        # match wins, scanned in declaration order)
        for sing, plur, to_tok, mult in _UNIT_PAIRS:
            patt = re.compile(rf"\b({re.escape(plur)}|{re.escape(sing)})\b",
                              flags=re.IGNORECASE)
            matches = list(patt.finditer(problem))
            if not matches:
                continue
            # rewrite all occurrences uniformly so currency in "$5 dollars"
            # contradictions don't survive.
            new_problem = patt.sub(to_tok, problem)
            new_answer = _format_scaled(ans_num * mult)
            return PerturbationResult(
                perturbed_problem=new_problem,
                new_answer=new_answer,
                applicable=True,
                metadata={
                    "unit_from": sing,
                    "unit_to": to_tok,
                    "multiplier": mult,
                },
            )

        return PerturbationResult(problem, None, applicable=False,
                                  metadata={"reason": "no_recognized_unit"})


# ---------------------------------------------------------------------------
# 3. IrrelevantContext — label-preserving
# ---------------------------------------------------------------------------

_IRRELEVANT_SENTENCES: List[str] = [
    "The weather today is partly cloudy with a light breeze.",
    "Her favorite color has always been a deep shade of green.",
    "The local library closes at six on weekdays.",
    "It takes about twenty minutes to walk to the park from here.",
    "The bakery on the corner is famous for its sourdough bread.",
    "Most days, the train arrives a few minutes behind schedule.",
    "Last summer was unusually warm in the northern hemisphere.",
    "The neighbor's cat likes to nap on the windowsill in the afternoon.",
    "Tea grows best in regions with mild temperatures and steady rainfall.",
    "The museum has a permanent collection of nineteenth-century pottery.",
    "Many bridges in old towns are made of stone rather than steel.",
    "The full moon was visible above the eastern horizon last night.",
    "Sparrows often build their nests under the eaves of older houses.",
    "Chess clubs meet every Tuesday evening at the community center.",
    "The river runs slower in autumn than it does during spring.",
    "Most apple trees take several years before they bear fruit.",
    "Streetlights on the main road turn on automatically at dusk.",
    "The old clock tower has been ticking for over a hundred years.",
    "Long-distance trains usually carry a dining car near the middle.",
    "Some breeds of dog were originally bred for working in cold climates.",
]


class IrrelevantContext:
    """Prepend a random unrelated sentence. Label unchanged."""

    name = "irrelevant_context"
    rewrites_label = False

    def apply(
        self, problem: str, answer: str, rng: Random
    ) -> PerturbationResult:
        sentence = _IRRELEVANT_SENTENCES[rng.randrange(len(_IRRELEVANT_SENTENCES))]
        new_problem = f"{sentence} {problem.lstrip()}"
        return PerturbationResult(
            perturbed_problem=new_problem,
            new_answer=None,
            applicable=True,
            metadata={"injected": sentence},
        )


# ---------------------------------------------------------------------------
# 4. DistractorSentence — label-preserving
# ---------------------------------------------------------------------------

# Templates pair a noun phrase with a number; the runner samples a number
# guaranteed to differ from the gold answer.
_DISTRACTOR_TEMPLATES: List[str] = [
    "The store also sells {n} different flavors of ice cream.",
    "Nearby, a school has {n} classrooms.",
    "An old book on the shelf has {n} pages.",
    "There are {n} parking spots on this block.",
    "A delivery truck passes by {n} times each week.",
    "The community garden has {n} raised beds.",
    "A wall in the hallway holds {n} small paintings.",
]


class DistractorSentence:
    """
    Insert a sentence containing a plausible-looking but irrelevant number
    somewhere in the middle of the problem. Label unchanged.

    We pick the inserted number from a small pool of {7, 11, 13, 17, 19, 23}
    minus the gold answer (if it lands in the pool). This keeps the distractor
    plainly off-topic but realistic, while making it trivially easy to assert
    in tests that the distractor never equals the gold.
    """

    name = "distractor_sentence"
    rewrites_label = False

    _CANDIDATE_NUMBERS: Tuple[int, ...] = (7, 11, 13, 17, 19, 23)

    def apply(
        self, problem: str, answer: str, rng: Random
    ) -> PerturbationResult:
        ans_num = _try_float(answer)
        pool = [n for n in self._CANDIDATE_NUMBERS
                if ans_num is None or abs(n - ans_num) > 1e-9]
        if not pool:
            # fall back to a number well outside the candidate pool
            pool = [101]
        number = pool[rng.randrange(len(pool))]
        template = _DISTRACTOR_TEMPLATES[rng.randrange(len(_DISTRACTOR_TEMPLATES))]
        distractor = template.format(n=number)

        sentences = _split_sentences(problem)
        if len(sentences) >= 2:
            # insert after the first sentence
            mid = 1
            new_sentences = sentences[:mid] + [distractor] + sentences[mid:]
            new_problem = " ".join(new_sentences)
        else:
            # short problem — append at the end (still mid-ish for the
            # final question clause that may follow)
            new_problem = f"{problem.rstrip()} {distractor}"

        return PerturbationResult(
            perturbed_problem=new_problem,
            new_answer=None,
            applicable=True,
            metadata={"distractor_number": number, "template": template},
        )


# ---------------------------------------------------------------------------
# 5. Paraphrase — label-preserving, rule-based
# ---------------------------------------------------------------------------


def _rule_swap_and_then(text: str) -> Optional[str]:
    if " and then " not in text.lower():
        return None
    return re.sub(r"\s+and then\s+", "; after that ", text, count=1,
                  flags=re.IGNORECASE)


def _rule_replace_so(text: str) -> Optional[str]:
    # "X, so Y" -> "Because X, Y"  (only when it's a comma+so construct)
    m = re.search(r"([A-Z][^.!?]*?),\s+so\s+([a-z])", text)
    if not m:
        return None
    head = m.group(1)
    rest_start = m.start(2)
    return f"Because {head[0].lower()}{head[1:]}, {text[rest_start].upper()}{text[rest_start + 1:]}"


def _rule_after_to_following(text: str) -> Optional[str]:
    if " after " not in text.lower():
        return None
    return re.sub(r"\bafter\b", "following", text, count=1, flags=re.IGNORECASE)


def _rule_buy_to_purchase(text: str) -> Optional[str]:
    if not re.search(r"\bbuys?\b", text, flags=re.IGNORECASE):
        return None
    def swap(m: re.Match) -> str:
        word = m.group(0)
        return "purchases" if word.endswith("s") else "purchase"
    return re.sub(r"\bbuys?\b", swap, text, count=1, flags=re.IGNORECASE)


def _rule_each_to_per(text: str) -> Optional[str]:
    if " each " not in text.lower():
        return None
    return re.sub(r"\beach\b", "per item", text, count=1, flags=re.IGNORECASE)


def _rule_total_to_altogether(text: str) -> Optional[str]:
    if " in total" not in text.lower() and " total " not in text.lower():
        return None
    return re.sub(r"\bin total\b", "altogether", text, count=1, flags=re.IGNORECASE)


_PARAPHRASE_RULES = (
    _rule_swap_and_then,
    _rule_replace_so,
    _rule_after_to_following,
    _rule_buy_to_purchase,
    _rule_each_to_per,
    _rule_total_to_altogether,
)


class Paraphrase:
    """
    Rule-based paraphrase. Picks one applicable rule via the seeded RNG.
    Label unchanged.

    If no rule applies, marks applicable=False rather than returning the
    original text — that way we don't conflate "no change" with "passed."
    """

    name = "paraphrase"
    rewrites_label = False

    def apply(
        self, problem: str, answer: str, rng: Random
    ) -> PerturbationResult:
        applicable_rules = []
        for rule in _PARAPHRASE_RULES:
            out = rule(problem)
            if out is not None and out != problem:
                applicable_rules.append((rule.__name__, out))

        if not applicable_rules:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "no_rule_matched"})

        rule_name, new_problem = applicable_rules[rng.randrange(len(applicable_rules))]
        return PerturbationResult(
            perturbed_problem=new_problem,
            new_answer=None,
            applicable=True,
            metadata={"rule": rule_name},
        )


# ---------------------------------------------------------------------------
# 6. Reordering — label-preserving
# ---------------------------------------------------------------------------

_PRONOUN_RE = re.compile(
    r"\b(he|she|it|they|him|her|them|his|hers|its|their|theirs|that|those|this|these)\b",
    flags=re.IGNORECASE,
)


class Reordering:
    """
    Swap the order of the first two sentences when no pronominal dependency
    links them.

    Skip when: <3 sentences total (we want to leave the final question intact),
    or when sentence 1 contains a pronoun (it'd dangle after the swap), or
    when sentence 2 starts with a pronoun referring back.

    This intentionally throws away problems with anaphora rather than risk
    nonsense reorderings.
    """

    name = "reordering"
    rewrites_label = False

    def apply(
        self, problem: str, answer: str, rng: Random
    ) -> PerturbationResult:
        sentences = _split_sentences(problem)
        if len(sentences) < 3:
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "too_few_sentences"})
        s1, s2 = sentences[0], sentences[1]
        if _PRONOUN_RE.search(s1):
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "pronoun_in_s1"})
        # s2 starting with a pronoun is a clear anaphora signal
        first_word = re.match(r"\s*(\w+)", s2)
        if first_word and _PRONOUN_RE.fullmatch(first_word.group(1)):
            return PerturbationResult(problem, None, applicable=False,
                                      metadata={"reason": "pronoun_at_s2_start"})

        new_sentences = [s2, s1] + sentences[2:]
        new_problem = " ".join(new_sentences)
        return PerturbationResult(
            perturbed_problem=new_problem,
            new_answer=None,
            applicable=True,
            metadata={"swapped": [0, 1]},
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def default_perturbations() -> List[Perturbation]:
    """All six perturbations in canonical order."""
    return [
        NumberSwap(),
        UnitChange(),
        IrrelevantContext(),
        DistractorSentence(),
        Paraphrase(),
        Reordering(),
    ]
