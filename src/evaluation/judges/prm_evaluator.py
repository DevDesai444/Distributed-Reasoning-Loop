"""
PRM-based evaluator for the agreement framework.

Wraps a trained :class:`ShortcutPRM` so it produces verdicts compatible with
the existing pair-report machinery. Each verdict is the inverse of the
shortcut probability — a high shortcut probability means "this trace looks
like a shortcut" which the framework treats as a REJECT.

The wrapper exposes the same surface as the other evaluators used by
``scripts/agreement_eval.py``: a ``score_batch`` that returns floats in
``[0, 1]`` and a ``verdicts`` helper that thresholds those floats into
ACCEPT/REJECT/UNCERTAIN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import JudgeVerdict, Verdict

logger = logging.getLogger(__name__)


DEFAULT_SHORTCUT_THRESHOLD = 0.5
PRM_PROMPT_VERSION = "shortcut-prm-v1"


@dataclass
class PRMEvaluatorConfig:
    """Light config for the PRM evaluator."""

    model_path: str = ""
    shortcut_threshold: float = DEFAULT_SHORTCUT_THRESHOLD
    extra: Dict[str, Any] = field(default_factory=dict)


class PRMEvaluator:
    """Use a trained shortcut PRM as a 4th evaluator in agreement runs."""

    judge_id = "shortcut-prm"
    prompt_version = PRM_PROMPT_VERSION

    def __init__(self, model: Any, *, config: Optional[PRMEvaluatorConfig] = None):
        self.model = model
        self.config = config or PRMEvaluatorConfig()

    @classmethod
    def from_path(
        cls,
        model_path: str | Path,
        *,
        shortcut_threshold: float = DEFAULT_SHORTCUT_THRESHOLD,
    ) -> "PRMEvaluator":
        try:
            from training.shortcut_prm import ShortcutPRM
        except ImportError:  # pragma: no cover - import path fallback
            from src.training.shortcut_prm import ShortcutPRM

        prm = ShortcutPRM.load(model_path)
        return cls(
            prm,
            config=PRMEvaluatorConfig(
                model_path=str(model_path),
                shortcut_threshold=shortcut_threshold,
            ),
        )

    # -------------------------------------------------------- scoring API

    def score_batch(self, traces: List[str]) -> List[float]:
        """Return the shortcut probability for each trace, in [0, 1]."""
        return self.model.score_batch(list(traces))

    def score(self, trace_text: str) -> float:
        return self.model.score(trace_text)

    def verdicts(
        self,
        traces: List[str],
        *,
        shortcut_threshold: Optional[float] = None,
    ) -> List[Optional[bool]]:
        """
        Convert shortcut probabilities to ACCEPT/REJECT verdicts.

        A trace with shortcut probability above the threshold is REJECTed.
        Borderline (probability == threshold) is treated as ACCEPT to stay
        consistent with the RM thresholding logic in ``agreement_eval``.
        """
        threshold = (
            shortcut_threshold
            if shortcut_threshold is not None
            else self.config.shortcut_threshold
        )
        scores = self.score_batch(traces)
        return [bool(score < threshold) for score in scores]

    def judge_verdicts(
        self,
        traces: List[str],
        *,
        shortcut_threshold: Optional[float] = None,
    ) -> List[JudgeVerdict]:
        """
        Return one :class:`JudgeVerdict` per trace, so the PRM column can be
        consumed by the same downstream code that consumes the LLM judge.
        """
        threshold = (
            shortcut_threshold
            if shortcut_threshold is not None
            else self.config.shortcut_threshold
        )
        scores = self.score_batch(traces)
        out: List[JudgeVerdict] = []
        for score in scores:
            if score >= threshold:
                verdict = Verdict.REJECT
                reason = "INCOMPLETE_REASONING"
            else:
                verdict = Verdict.ACCEPT
                reason = "NONE"
            out.append(
                JudgeVerdict(
                    verdict=verdict,
                    confidence=float(abs(score - threshold) * 2),
                    rationale=f"shortcut probability {score:.3f} vs threshold {threshold:.2f}",
                    reason_code=reason,
                    raw_response="",
                    judge_id=self.judge_id,
                    prompt_version=self.prompt_version,
                    metadata={"shortcut_probability": score, "threshold": threshold},
                )
            )
        return out
