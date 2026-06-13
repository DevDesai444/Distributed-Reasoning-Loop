"""
Robustness evaluation: perturbation transformations + retention runner.

The six perturbations live in `perturbations.py` and the retention runner
lives in `runner.py`. The CLI wrapper is `scripts/robustness_eval.py` and is
mounted under `python main.py eval robust ...`.
"""

from .perturbations import (
    DistractorSentence,
    IrrelevantContext,
    NumberSwap,
    Paraphrase,
    Perturbation,
    PerturbationResult,
    Reordering,
    UnitChange,
    default_perturbations,
)
from .runner import (
    PerturbationOutcome,
    PerturbationStats,
    RobustnessReport,
    evaluate_robustness,
    verdicts_to_stats,
)

__all__ = [
    "DistractorSentence",
    "IrrelevantContext",
    "NumberSwap",
    "Paraphrase",
    "Perturbation",
    "PerturbationResult",
    "Reordering",
    "UnitChange",
    "default_perturbations",
    "PerturbationOutcome",
    "PerturbationStats",
    "RobustnessReport",
    "evaluate_robustness",
    "verdicts_to_stats",
]
