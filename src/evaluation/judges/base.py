"""
Judge interface and verdict dataclass shared by every judge variant.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .prompts import (
    REASON_CODES,
    VERDICT_ACCEPT,
    VERDICT_REJECT,
    VERDICT_UNCERTAIN,
)


class Verdict(str, Enum):
    ACCEPT = VERDICT_ACCEPT
    REJECT = VERDICT_REJECT
    UNCERTAIN = VERDICT_UNCERTAIN

    @classmethod
    def from_string(cls, raw: str) -> "Verdict":
        token = (raw or "").strip().upper()
        if token in {v.value for v in cls}:
            return cls(token)
        # parser tolerated forms — model sometimes adds punctuation
        token = re.sub(r"[^A-Z]", "", token)
        if token in {v.value for v in cls}:
            return cls(token)
        return cls.UNCERTAIN


@dataclass
class JudgeVerdict:
    """One judge's call on one trace."""

    verdict: Verdict
    confidence: float
    rationale: str
    reason_code: str = "NONE"
    raw_response: str = ""
    judge_id: str = ""
    prompt_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["verdict"] = self.verdict.value
        return payload

    @property
    def as_bool(self) -> Optional[bool]:
        """Return True/False for ACCEPT/REJECT, None for UNCERTAIN."""
        if self.verdict == Verdict.ACCEPT:
            return True
        if self.verdict == Verdict.REJECT:
            return False
        return None


def parse_judge_response(text: str) -> Dict[str, Any]:
    """
    Pull out VERDICT / CONFIDENCE / REASON_CODE / RATIONALE fields from a
    rubric reply. Tolerant of stray whitespace and the occasional bullet point.
    """
    if not text:
        return {
            "verdict": Verdict.UNCERTAIN,
            "confidence": 0.0,
            "reason_code": "OTHER",
            "rationale": "",
        }

    fields: Dict[str, str] = {}
    patterns = {
        "verdict": r"VERDICT\s*[:\-]?\s*([A-Za-z]+)",
        "confidence": r"CONFIDENCE\s*[:\-]?\s*([0-9.]+)",
        "reason_code": r"REASON_CODE\s*[:\-]?\s*([A-Z_]+)",
        "rationale": r"RATIONALE\s*[:\-]?\s*(.+?)(?:\n[A-Z_]+\s*[:\-]|\Z)",
    }

    for key, pat in patterns.items():
        match = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            fields[key] = match.group(1).strip()

    verdict = Verdict.from_string(fields.get("verdict", "UNCERTAIN"))
    try:
        confidence = float(fields.get("confidence", "0").strip())
    except ValueError:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason_raw = fields.get("reason_code", "NONE").upper().strip()
    if reason_raw not in REASON_CODES and reason_raw != "NONE":
        reason_raw = "OTHER"
    if verdict == Verdict.ACCEPT:
        reason_raw = "NONE"

    rationale = fields.get("rationale", "").strip()
    # trim runaway rationale; model sometimes keeps going
    if len(rationale) > 800:
        rationale = rationale[:800].rstrip() + "..."

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason_code": reason_raw,
        "rationale": rationale,
    }


class Judge(ABC):
    """Common surface for any judge — single, pairwise, or panel."""

    judge_id: str = "abstract"

    @abstractmethod
    def judge(
        self,
        *,
        problem: str,
        response: str,
        reference_answer: Optional[str] = None,
        problem_type: str = "math",
    ) -> JudgeVerdict:
        """Evaluate a single (problem, response) pair."""

    def judge_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[JudgeVerdict]:
        """
        Default batch implementation calls judge() in sequence.
        Subclasses with native batching should override.
        """
        return [self.judge(**item) for item in items]
