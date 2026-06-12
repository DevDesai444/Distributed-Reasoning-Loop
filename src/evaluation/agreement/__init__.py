"""
Agreement metrics + report types for cross-evaluator comparison.
"""

from .difficulty_binner import (
    bin_for_problem,
    bin_for_step_count,
    bin_problems,
    step_count_from_solution,
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
]
