"""
Agreement report — bundles the metric outputs into a serializable artifact and
can render itself as JSON or Markdown.

We keep the report itself dumb: the CLI assembles the data, the report just
holds it and emits text.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .metrics import (
    agreement_rate,
    bootstrap_ci,
    cohens_kappa,
    confusion_matrix,
)


@dataclass
class PairReport:
    """Metrics for one (rater_a, rater_b) comparison."""

    rater_a: str
    rater_b: str
    n: int
    agreement: float
    kappa: float
    kappa_ci_low: float
    kappa_ci_high: float
    confusion: Dict[str, int]
    per_bin: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgreementReport:
    """Top-level container for an agreement run."""

    run_id: str
    model: str
    benchmark: str
    n_problems: int
    n_traces: int
    judge_model: str
    judge_prompt_version: str
    pair_reports: List[PairReport] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "benchmark": self.benchmark,
            "n_problems": self.n_problems,
            "n_traces": self.n_traces,
            "judge_model": self.judge_model,
            "judge_prompt_version": self.judge_prompt_version,
            "pair_reports": [p.to_dict() for p in self.pair_reports],
            "metadata": self.metadata,
        }

    def render_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def render_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# Agreement Report — `{self.run_id}`")
        lines.append("")
        lines.append(f"- Model under test: `{self.model}`")
        lines.append(f"- Benchmark: `{self.benchmark}`")
        lines.append(f"- Problems: {self.n_problems}, traces: {self.n_traces}")
        lines.append(f"- Judge model: `{self.judge_model}` (prompt {self.judge_prompt_version})")
        lines.append("")
        lines.append("## Pairwise agreement")
        lines.append("")
        lines.append("| Pair | N | Agreement | Kappa | Kappa 95% CI |")
        lines.append("|---|---:|---:|---:|---|")
        for pair in self.pair_reports:
            ci = f"[{pair.kappa_ci_low:.3f}, {pair.kappa_ci_high:.3f}]"
            lines.append(
                f"| {pair.rater_a} vs {pair.rater_b} | {pair.n} | "
                f"{pair.agreement:.3f} | {pair.kappa:.3f} | {ci} |"
            )
        lines.append("")

        for pair in self.pair_reports:
            lines.append(f"### {pair.rater_a} vs {pair.rater_b} — confusion")
            lines.append("")
            c = pair.confusion
            lines.append("| | B accept | B reject |")
            lines.append("|---|---:|---:|")
            lines.append(f"| A accept | {c['tp']} | {c['fn']} |")
            lines.append(f"| A reject | {c['fp']} | {c['tn']} |")
            lines.append(f"(dropped uncertain pairs: {c.get('dropped_uncertain', 0)})")
            lines.append("")

            if pair.per_bin:
                lines.append(f"#### Per-difficulty bin — {pair.rater_a} vs {pair.rater_b}")
                lines.append("")
                lines.append("| Bin | N | Agreement | Kappa |")
                lines.append("|---|---:|---:|---:|")
                for bin_name, stats in pair.per_bin.items():
                    lines.append(
                        f"| {bin_name} | {stats.get('n', 0)} | "
                        f"{stats.get('agreement', 0.0):.3f} | "
                        f"{stats.get('kappa', 0.0):.3f} |"
                    )
                lines.append("")

        if self.metadata:
            lines.append("## Metadata")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(self.metadata, indent=2))
            lines.append("```")

        return "\n".join(lines)


def build_pair_report(
    rater_a: str,
    rater_b: str,
    verdicts_a: List[Optional[bool]],
    verdicts_b: List[Optional[bool]],
    *,
    bins: Optional[Dict[str, List[int]]] = None,
    n_bootstrap: int = 500,
) -> PairReport:
    """
    Compute overall + per-bin metrics for one comparison.

    `bins` is an optional map from bin_name to indices (into verdicts_*).
    """
    overall_kappa = cohens_kappa(verdicts_a, verdicts_b)
    overall_agreement = agreement_rate(verdicts_a, verdicts_b)
    confusion = confusion_matrix(verdicts_a, verdicts_b)
    ci_low, ci_high = bootstrap_ci(
        cohens_kappa,
        verdicts_a,
        verdicts_b,
        n_resamples=n_bootstrap,
    )

    per_bin: Dict[str, Dict[str, Any]] = {}
    if bins:
        for bin_name, indices in bins.items():
            if not indices:
                continue
            sub_a = [verdicts_a[i] for i in indices if i < len(verdicts_a)]
            sub_b = [verdicts_b[i] for i in indices if i < len(verdicts_b)]
            per_bin[bin_name] = {
                "n": len(sub_a),
                "agreement": agreement_rate(sub_a, sub_b),
                "kappa": cohens_kappa(sub_a, sub_b),
            }

    n_valid = sum(1 for x, y in zip(verdicts_a, verdicts_b) if x is not None and y is not None)
    return PairReport(
        rater_a=rater_a,
        rater_b=rater_b,
        n=n_valid,
        agreement=overall_agreement,
        kappa=overall_kappa,
        kappa_ci_low=ci_low,
        kappa_ci_high=ci_high,
        confusion=confusion,
        per_bin=per_bin,
    )
