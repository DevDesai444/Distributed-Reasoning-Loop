#!/usr/bin/env python3
"""
N-gram contamination report runner.

Entry point behind ``python main.py eval contamination ...``. Loads train/test
splits of a benchmark via the existing dataset loaders, runs 13-gram overlap
analysis in both directions, and emits a machine-readable JSON report plus a
human-readable Markdown summary.

If ``--synthetic-pairs`` is given, also checks the chosen+rejected traces of
the synthetic pairs against the benchmark test split. Catches the case where
the generator memorised test problems and surfaced them in training data.

If ``--external`` is given, also checks an arbitrary JSONL corpus (each line a
JSON object with at least a ``text`` or ``problem`` field) against the test
split. Useful when the user has the pretraining set on hand.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.contamination.overlap import (
    ContaminationReport,
    analyze_overlap,
    report_to_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


BENCHMARK_CHOICES = ["gsm8k", "math", "humaneval", "mbpp"]


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("contam_%Y%m%dT%H%M%SZ")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="N-gram contamination analysis between benchmark splits.",
    )
    p.add_argument("--benchmark", type=str, default="gsm8k", choices=BENCHMARK_CHOICES)
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to write contamination_report.json and report.md.")
    p.add_argument("--external", type=str, default=None,
                   help="Optional path to a JSONL corpus to also compare against test.")
    p.add_argument("--synthetic-pairs", type=str, default=None,
                   help="Optional path to data/dpo_pairs.jsonl to check generator leakage.")
    p.add_argument("--n", type=int, default=13, help="N-gram size. Default 13.")
    p.add_argument("--high-threshold", type=int, default=3,
                   help="A needle counts as high-overlap if it has >= this many matching n-grams.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of worst-offender matches to include per pass.")
    p.add_argument("--sample-corpus", type=str, default=None,
                   help="Optional path to a sample JSONL with 'split' field, used as fallback "
                        "when the HuggingFace loader cannot reach the network. The file should "
                        "contain at least 'split', 'id', and 'problem' fields per line.")
    p.add_argument("--force-sample-corpus", action="store_true",
                   help="Use the sample corpus even if the real loader would succeed. "
                        "Useful for deterministic tests and for re-running a known fallback.")
    p.add_argument("--max-haystack", type=int, default=0,
                   help="Cap the haystack size at this many problems (0 = no cap). "
                        "Use for fast smoke runs.")
    p.add_argument("--max-needles", type=int, default=0,
                   help="Cap the needle corpus at this many problems (0 = no cap).")
    return p


def _load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{i}: {exc}") from exc
    return out


@dataclass(frozen=True)
class _Splits:
    train: List[dict]
    test: List[dict]
    source: str  # "loader" or "sample-fallback"


def _normalise_problem_records(rows: Iterable[dict], split: str) -> List[dict]:
    """Coerce loader/JSONL rows into the {id, text} schema the analyser expects."""
    out: List[dict] = []
    for i, row in enumerate(rows):
        rid = row.get("id") or row.get("task_id") or f"{split}_{i}"
        text = row.get("problem") or row.get("text") or row.get("question") or row.get("prompt")
        if text is None:
            continue
        out.append({"id": str(rid), "text": str(text)})
    return out


def _load_via_arrow_cache(benchmark: str) -> Optional[_Splits]:
    """Fast path for the bundled GSM8K arrow cache. Returns None on miss.

    The default ``GSM8KLoader.load`` calls ``datasets.load_dataset`` which on
    certain ``datasets`` versions fails with ``LocalFileSystem is not supported``
    even when the data is already in ``~/.cache/huggingface/datasets``. We
    sidestep that by opening the arrow files directly.
    """
    if benchmark != "gsm8k":
        return None
    base = Path.home() / ".cache" / "huggingface" / "datasets" / "openai___gsm8k" / "main"
    if not base.exists():
        return None
    # Cache layout is .../main/<version>/<hash>/gsm8k-{split}.arrow. Use rglob
    # so we tolerate version/hash drift across `datasets` releases.
    train_files = sorted(base.rglob("gsm8k-train.arrow"))
    test_files = sorted(base.rglob("gsm8k-test.arrow"))
    if not train_files or not test_files:
        return None
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        return None

    def _read(path: Path, split: str) -> List[dict]:
        ds = Dataset.from_file(str(path))
        rows = []
        for idx, item in enumerate(ds):
            rows.append({"id": f"gsm8k_{split}_{idx}", "text": item["question"]})
        return rows

    train = _read(train_files[0], "train")
    test = _read(test_files[0], "test")
    return _Splits(train=train, test=test, source="loader")


def _load_via_dataset_loader(benchmark: str) -> Optional[_Splits]:
    """Try the canonical project loader. Returns None if network/import fails."""
    try:
        from src.data_generator.dataset_loader import get_loader
    except ImportError as exc:
        logger.warning("dataset_loader import failed: %s", exc)
        return None

    try:
        if benchmark == "gsm8k":
            train_problems = get_loader("gsm8k", split="train").load()
            test_problems = get_loader("gsm8k", split="test").load()
        elif benchmark == "math":
            train_problems = get_loader("math", split="train").load()
            test_problems = get_loader("math", split="test").load()
        elif benchmark == "humaneval":
            # HumanEval has no canonical train split. We treat its test split as
            # both haystack and needle so the diagonal pass at least surfaces
            # internal near-duplicates, then leave train pairs empty.
            test_problems = get_loader("humaneval").load()
            train_problems = []
        elif benchmark == "mbpp":
            train_problems = get_loader("mbpp", split="train").load()
            test_problems = get_loader("mbpp", split="test").load()
        else:
            return None
    except Exception as exc:  # network errors, missing files, etc.
        logger.warning("dataset_loader failed for %s: %s", benchmark, exc)
        return None

    def _to_dict(p) -> dict:
        return {"id": p.id, "text": p.problem}

    return _Splits(
        train=[_to_dict(p) for p in train_problems],
        test=[_to_dict(p) for p in test_problems],
        source="loader",
    )


def _load_sample_fallback(benchmark: str, path: Path) -> _Splits:
    """Load the bundled sample-corpus fallback."""
    rows = _load_jsonl(path)
    train_rows = [r for r in rows if r.get("benchmark") == benchmark and r.get("split") == "train"]
    test_rows = [r for r in rows if r.get("benchmark") == benchmark and r.get("split") == "test"]
    train = _normalise_problem_records(train_rows, "train")
    test = _normalise_problem_records(test_rows, "test")
    return _Splits(train=train, test=test, source="sample-fallback")


def _load_benchmark_splits(
    benchmark: str,
    sample_corpus: Optional[Path],
    prefer_sample: bool = False,
) -> _Splits:
    """Best-effort loader. Tries arrow cache → dataset_loader → sample fallback.

    If ``prefer_sample`` is set and a sample corpus is supplied, we use it
    directly without consulting the real loaders. That's how tests pin the
    data source — the production CLI does not set this flag.
    """
    if prefer_sample and sample_corpus is not None and sample_corpus.exists():
        logger.info("using sample corpus at %s (forced)", sample_corpus)
        return _load_sample_fallback(benchmark, sample_corpus)

    splits = _load_via_arrow_cache(benchmark)
    if splits is not None and (splits.train or splits.test):
        logger.info("loaded %s from arrow cache: %d train / %d test",
                    benchmark, len(splits.train), len(splits.test))
        return splits

    splits = _load_via_dataset_loader(benchmark)
    if splits is not None and (splits.train or splits.test):
        logger.info("loaded %s via dataset_loader: %d train / %d test",
                    benchmark, len(splits.train), len(splits.test))
        return splits

    if sample_corpus is not None and sample_corpus.exists():
        logger.warning("falling back to sample corpus at %s", sample_corpus)
        return _load_sample_fallback(benchmark, sample_corpus)

    raise RuntimeError(
        f"could not load {benchmark} splits: no arrow cache, dataset_loader failed, "
        f"and no --sample-corpus supplied"
    )


def _load_external(path: Path) -> List[dict]:
    rows = _load_jsonl(path)
    return _normalise_problem_records(rows, "external")


def _load_synthetic_pairs(path: Path) -> List[dict]:
    """Extract chosen+rejected traces from a v1-envelope dpo_pairs.jsonl."""
    rows = _load_jsonl(path)
    out: List[dict] = []
    for i, row in enumerate(rows):
        # v1 envelope: {pair: {problem_id, chosen, rejected, ...}, ...}
        pair = row.get("pair", row)
        pid = pair.get("problem_id") or f"pair_{i}"
        for kind in ("chosen", "rejected"):
            text = pair.get(kind)
            if text:
                out.append({"id": f"{pid}_{kind}", "text": text})
    return out


def _maybe_cap(rows: List[dict], cap: int) -> List[dict]:
    if cap and cap > 0 and len(rows) > cap:
        return rows[:cap]
    return rows


def _write_json(reports: List[ContaminationReport], meta: dict, path: Path) -> None:
    payload = {
        "schema_version": "contamination/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "reports": [report_to_dict(r) for r in reports],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _format_markdown(reports: List[ContaminationReport], meta: dict) -> str:
    lines: List[str] = []
    lines.append("# Contamination Report")
    lines.append("")
    lines.append(f"- benchmark: `{meta['benchmark']}`")
    lines.append(f"- data source: `{meta['data_source']}`")
    lines.append(f"- n-gram size: `{meta['n']}`")
    lines.append(f"- high-overlap threshold: `>= {meta['high_threshold']} matching n-grams`")
    lines.append(f"- generated: `{meta['generated_at']}`")
    if meta.get("notes"):
        lines.append("")
        lines.append(f"> {meta['notes']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Direction | Haystack size | Needle size | Any overlap | High overlap |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in reports:
        lines.append(
            f"| {r.needle_name} in {r.haystack_name} "
            f"| {r.haystack_total_problems} "
            f"| {r.needle_total_problems} "
            f"| {r.needle_problems_with_any_overlap} ({r.overlap_rate_any:.2%}) "
            f"| {r.needle_problems_with_high_overlap} ({r.overlap_rate_high:.2%}) |"
        )
    lines.append("")
    lines.append("## Top matches")
    lines.append("")
    for r in reports:
        lines.append(f"### {r.needle_name} in {r.haystack_name}")
        lines.append("")
        if not r.top_matches:
            lines.append("_no overlapping needles_")
            lines.append("")
            continue
        for m in r.top_matches:
            preview = m.needle_text.replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            lines.append(f"- `{m.needle_id}` — {m.overlap_count} matching n-grams")
            lines.append(f"  - text: {preview}")
            if m.overlapping_ngrams:
                first = " ".join(m.overlapping_ngrams[0])
                lines.append(f"  - example n-gram: `{first}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_corpus = Path(args.sample_corpus) if args.sample_corpus else None
    splits = _load_benchmark_splits(
        args.benchmark, sample_corpus, prefer_sample=args.force_sample_corpus,
    )

    train = _maybe_cap(splits.train, args.max_haystack)
    test = _maybe_cap(splits.test, args.max_needles)

    reports: List[ContaminationReport] = []

    # Standard symmetric passes.
    if train and test:
        reports.append(
            analyze_overlap(
                haystack=train, needles=test, n=args.n,
                high_threshold=args.high_threshold, top_k=args.top_k,
                haystack_name=f"{args.benchmark}/train", needle_name=f"{args.benchmark}/test",
            )
        )
        reports.append(
            analyze_overlap(
                haystack=test, needles=train, n=args.n,
                high_threshold=args.high_threshold, top_k=args.top_k,
                haystack_name=f"{args.benchmark}/test", needle_name=f"{args.benchmark}/train",
            )
        )
    elif test:
        # HumanEval-style: only a test split exists. Run test-against-test to
        # surface internal near-duplicates.
        logger.info("no train split for %s; running test-vs-test diagonal pass", args.benchmark)
        reports.append(
            analyze_overlap(
                haystack=test, needles=test, n=args.n,
                high_threshold=args.high_threshold, top_k=args.top_k,
                haystack_name=f"{args.benchmark}/test", needle_name=f"{args.benchmark}/test",
            )
        )

    # External-corpus pass.
    if args.external:
        ext_path = Path(args.external)
        external = _load_external(ext_path)
        if external and test:
            reports.append(
                analyze_overlap(
                    haystack=external, needles=test, n=args.n,
                    high_threshold=args.high_threshold, top_k=args.top_k,
                    haystack_name=f"external:{ext_path.name}",
                    needle_name=f"{args.benchmark}/test",
                )
            )

    # Synthetic-pairs pass.
    if args.synthetic_pairs:
        pairs_path = Path(args.synthetic_pairs)
        pair_traces = _load_synthetic_pairs(pairs_path)
        if pair_traces and test:
            reports.append(
                analyze_overlap(
                    haystack=pair_traces, needles=test, n=args.n,
                    high_threshold=args.high_threshold, top_k=args.top_k,
                    haystack_name=f"synthetic_pairs:{pairs_path.name}",
                    needle_name=f"{args.benchmark}/test",
                )
            )

    notes = None
    if splits.source == "sample-fallback":
        notes = ("Numbers below are based on a small bundled sample corpus, not the "
                 "full benchmark splits. Re-run with the HuggingFace cache populated "
                 "to get the canonical report.")

    meta = {
        "benchmark": args.benchmark,
        "data_source": splits.source,
        "n": args.n,
        "high_threshold": args.high_threshold,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }

    json_path = output_dir / "contamination_report.json"
    md_path = output_dir / "report.md"
    _write_json(reports, meta, json_path)
    md_path.write_text(_format_markdown(reports, meta))

    logger.info("wrote %s", json_path)
    logger.info("wrote %s", md_path)
    for r in reports:
        logger.info(
            "%s in %s: any=%d/%d (%.2f%%), high=%d/%d (%.2f%%)",
            r.needle_name, r.haystack_name,
            r.needle_problems_with_any_overlap, r.needle_total_problems,
            100 * r.overlap_rate_any,
            r.needle_problems_with_high_overlap, r.needle_total_problems,
            100 * r.overlap_rate_high,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
