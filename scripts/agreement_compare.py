#!/usr/bin/env python3
"""
Cross-benchmark agreement comparison.

Reads two or more agreement receipts (``agreement_v1*.json``) and emits
a one-table markdown summary of judge reliability across benchmarks.

The table is the headline artifact for the "Three-way agreement"
README subsection: one row per benchmark, columns for n_traces,
verifier-vs-judge Cohen's kappa with 95% CI, and shortcut rate.

Example::

    python scripts/agreement_compare.py \\
        --receipt receipts/agreement_v1.json \\
        --receipt receipts/agreement_v1_math.json \\
        --receipt receipts/agreement_v1_humaneval.json \\
        --output eval_receipts/agreement_cross_benchmark.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.receipt_schemas import AgreementReceipt, load_receipt  # noqa: E402


logger = logging.getLogger("agreement_compare")


VERIFIER_NOTES = {
    "gsm8k": "numeric equivalence",
    "math": "symbolic equivalence",
    "humaneval": "sandbox execution",
}


def _format_kappa(k: Optional[float]) -> str:
    return f"{k:.3f}" if isinstance(k, (int, float)) else "n/a"


def _format_ci(ci: List[Optional[float]]) -> str:
    if not ci or len(ci) != 2:
        return "n/a"
    lo, hi = ci
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:.3f}, {hi:.3f}]"


def _format_rate(rate: Optional[float]) -> str:
    if rate is None:
        return "n/a"
    return f"{rate * 100:.1f}%"


def _find_pair(receipt: AgreementReceipt, name_a: str, name_b: str):
    """Return the verifier vs judge pair from a receipt, or None."""
    targets = {
        f"{name_a}_vs_{name_b}",
        f"{name_b}_vs_{name_a}",
    }
    for pair in receipt.pairs:
        if pair.pair_name in targets:
            return pair
    return None


def render_comparison(receipts: List[AgreementReceipt]) -> str:
    """Build the cross-benchmark markdown table."""
    if len(receipts) < 2:
        raise ValueError("agreement_compare needs at least two receipts")

    lines: List[str] = []
    lines.append("# Judge Reliability Across Benchmarks")
    lines.append("")
    lines.append(
        "| Benchmark | n_traces | "
        "kappa (verifier vs judge) | 95% CI | Shortcut rate |"
    )
    lines.append("|---|---:|---:|---|---:|")

    for receipt in receipts:
        pair = _find_pair(receipt, "verifier", "judge")
        kappa = pair.cohens_kappa if pair else None
        ci = pair.kappa_ci_95 if pair else [None, None]
        rate = receipt.shortcut_bucket.shortcut_rate
        lines.append(
            f"| {receipt.benchmark} | {receipt.n_traces} | "
            f"{_format_kappa(kappa)} | {_format_ci(ci)} | {_format_rate(rate)} |"
        )

    lines.append("")
    lines.append("Notes:")
    for receipt in receipts:
        base = str(receipt.benchmark).split("-")[0].lower()
        note = VERIFIER_NOTES.get(base, "see receipt notes")
        lines.append(f"- {receipt.benchmark} verifier: {note}")
    lines.append("")
    lines.append(
        "Receipts: " + ", ".join(f"`{r.run_id}`" for r in receipts)
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt", action="append", required=True,
                   help="Path to an agreement receipt. Pass at least twice.")
    p.add_argument("--output", type=str, required=True,
                   help="Markdown output path.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _build_parser().parse_args(argv)
    paths = [Path(p) for p in args.receipt]
    receipts: List[AgreementReceipt] = []
    for path in paths:
        receipt = load_receipt(path)
        if not isinstance(receipt, AgreementReceipt):
            raise ValueError(
                f"{path} is not an agreement receipt (kind={receipt.kind!r})"
            )
        receipt.validate()
        receipts.append(receipt)

    markdown = render_comparison(receipts)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    logger.info("wrote %s", out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
