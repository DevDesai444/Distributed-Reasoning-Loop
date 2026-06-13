"""
Run-level statistics for the synthetic data pipeline.

Collects counts and a small percentile sketch over quality scores, then
emits a `data/stats.json` summary at end-of-run. Shape is documented in
the phase plan.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .provenance import PROVENANCE_SCHEMA_VERSION, utc_now_iso


@dataclass
class BackendCounters:
    generations: int = 0
    accepts: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {"generations": self.generations, "accepts": self.accepts}


@dataclass
class RunStats:
    """Mutable aggregator. Call `record_*` while generating, then `to_dict()`."""

    schema_version: str = PROVENANCE_SCHEMA_VERSION
    total_problems_attempted: int = 0
    total_generations: int = 0
    verifier_accepts: int = 0
    verifier_rejects: int = 0
    pair_candidates: int = 0
    unique_pairs_after_dedup: int = 0
    per_backend: Dict[str, BackendCounters] = field(default_factory=dict)
    _quality_scores: List[float] = field(default_factory=list)

    def record_problem_attempt(self) -> None:
        self.total_problems_attempted += 1

    def record_generation(self, backend: str, accepted: bool) -> None:
        self.total_generations += 1
        if accepted:
            self.verifier_accepts += 1
        else:
            self.verifier_rejects += 1
        counters = self.per_backend.setdefault(backend, BackendCounters())
        counters.generations += 1
        if accepted:
            counters.accepts += 1

    def record_pair_candidate(self) -> None:
        self.pair_candidates += 1

    def record_unique_pair(self, quality_score: Optional[float]) -> None:
        self.unique_pairs_after_dedup += 1
        if quality_score is not None:
            bisect.insort(self._quality_scores, float(quality_score))

    def _percentile(self, p: float) -> Optional[float]:
        if not self._quality_scores:
            return None
        # nearest-rank percentile, clamped
        idx = max(0, min(len(self._quality_scores) - 1, int(round(p * (len(self._quality_scores) - 1)))))
        return self._quality_scores[idx]

    def to_dict(self) -> Dict[str, Any]:
        accept_rate = (
            self.verifier_accepts / self.total_generations
            if self.total_generations
            else 0.0
        )
        dedup_ratio = (
            self.unique_pairs_after_dedup / self.pair_candidates
            if self.pair_candidates
            else 0.0
        )
        return {
            "schema_version": self.schema_version,
            "total_problems_attempted": self.total_problems_attempted,
            "total_generations": self.total_generations,
            "verifier_accepts": self.verifier_accepts,
            "verifier_rejects": self.verifier_rejects,
            "verifier_accept_rate": round(accept_rate, 4),
            "pair_candidates": self.pair_candidates,
            "unique_pairs_after_dedup": self.unique_pairs_after_dedup,
            "dedup_ratio": round(dedup_ratio, 4),
            "per_backend": {k: v.to_dict() for k, v in self.per_backend.items()},
            "quality_score_distribution": {
                "p50": self._percentile(0.50),
                "p90": self._percentile(0.90),
                "p99": self._percentile(0.99),
            },
            "completed_at": utc_now_iso(),
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


__all__ = ["BackendCounters", "RunStats"]
