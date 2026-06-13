"""
Agreement metrics + report types for cross-evaluator comparison.
"""

from .difficulty_binner import (
    UNIFORM_BIN,
    UniformBinner,
    bin_for_problem,
    bin_for_step_count,
    bin_problems,
    step_count_from_solution,
    uniform_bin,
    uniform_bin_problems,
)
from .math_difficulty_binner import (
    ALL_BINS as MATH_LEVEL_BINS,
    MathDifficultyBinner,
    UNKNOWN_BIN as MATH_UNKNOWN_BIN,
    bin_for_level as math_bin_for_level,
    bin_for_problem as math_bin_for_problem,
    bin_problems as math_bin_problems,
    parse_level as math_parse_level,
)
from .metrics import (
    agreement_rate,
    bootstrap_ci,
    cohens_kappa,
    confusion_matrix,
)
from .report import AgreementReport, PairReport, build_pair_report

__all__ = [
    "agreement_rate",
    "bootstrap_ci",
    "cohens_kappa",
    "confusion_matrix",
    "AgreementReport",
    "PairReport",
    "build_pair_report",
    "step_count_from_solution",
    "bin_for_step_count",
    "bin_for_problem",
    "bin_problems",
    "UNIFORM_BIN",
    "uniform_bin",
    "uniform_bin_problems",
    "UniformBinner",
    "MathDifficultyBinner",
    "MATH_LEVEL_BINS",
    "MATH_UNKNOWN_BIN",
    "math_bin_for_level",
    "math_bin_for_problem",
    "math_bin_problems",
    "math_parse_level",
]
