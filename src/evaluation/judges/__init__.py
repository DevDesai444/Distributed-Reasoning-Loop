"""
LLM-as-judge subsystem.

Three concrete judges (single / pairwise / panel) sharing a common verdict
type and rubric-prompt set. Bias-control utilities live alongside.
"""

from .base import Judge, JudgeVerdict, Verdict, parse_judge_response
from .bias_controls import (
    PositionSwapResult,
    SelfPreferenceResult,
    length_bias_probe,
    position_swap_stability,
    self_preference_delta,
    truncate_to_min_length,
)
from .pairwise_judge import PairwiseJudge, PairwiseVerdict
from .panel_judge import PanelConfig, PanelJudge
from .prm_evaluator import PRMEvaluator, PRMEvaluatorConfig
from .prompts import (
    CODE_RUBRIC_V1,
    MATH_RUBRIC_V1,
    PAIRWISE_RUBRIC_V1,
    PROMPT_VERSIONS,
    REASON_CODES,
    get_rubric,
)
from .single_model_judge import DEFAULT_JUDGE_MODEL, JudgeConfig, SingleModelJudge

__all__ = [
    "Judge",
    "JudgeVerdict",
    "Verdict",
    "parse_judge_response",
    "SingleModelJudge",
    "JudgeConfig",
    "DEFAULT_JUDGE_MODEL",
    "PairwiseJudge",
    "PairwiseVerdict",
    "PanelJudge",
    "PanelConfig",
    "MATH_RUBRIC_V1",
    "CODE_RUBRIC_V1",
    "PAIRWISE_RUBRIC_V1",
    "PROMPT_VERSIONS",
    "REASON_CODES",
    "get_rubric",
    "PositionSwapResult",
    "SelfPreferenceResult",
    "position_swap_stability",
    "self_preference_delta",
    "truncate_to_min_length",
    "length_bias_probe",
    "PRMEvaluator",
    "PRMEvaluatorConfig",
]
