#!/usr/bin/env python3
"""
CPU-only smoke run of the v1 synthetic pipeline emission path.

This does NOT call the LLM. It feeds canned reasoning traces through the
provenance + quality_score + checkpointer machinery so the artifacts in
`samples/synthetic_smoke/` exercise the full envelope without burning
GPU time.

The full pipeline is invoked by:

    python main.py generate --subset-size 50 --target-pairs 200 --with-provenance

Run that on a machine with the teacher model and GPU available.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import List, Tuple

# Allow direct execution.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_generator.checkpoint import PairCheckpointer  # noqa: E402
from data_generator.provenance import build_provenance  # noqa: E402
from data_generator.quality_score import score_trace  # noqa: E402
from data_generator.stats import RunStats  # noqa: E402


logger = logging.getLogger("smoke_synthetic_v1")


# A small bank of (problem, correct_trace, wrong_trace) tuples. Each
# correct trace embeds inline `<<a op b = c>>` blocks so the step-coherence
# component exercises a real signal.
_BANK: List[Tuple[str, str, str, str, str]] = [
    (
        "smoke_0001",
        "Alice has 12 apples and gives 3 to Bob. How many does she have left?",
        "Start with 12 apples. Give 3 away: <<12-3=9>>. Alice has 9 apples.",
        "Start with 12 apples. Give 3 away: <<12-3=10>>. Alice has 10 apples.",
        "9",
    ),
    (
        "smoke_0002",
        "A train moves at 60 mph for 2.5 hours. How far does it travel?",
        "Distance is speed times time: <<60*2.5=150>>. The train travels 150 miles.",
        "Distance is speed plus time: <<60+2.5=62.5>>. The train travels 62.5 miles.",
        "150",
    ),
    (
        "smoke_0003",
        "A rectangle has width 4 and height 7. What is its area?",
        "Area equals width times height: <<4*7=28>>. The area is 28.",
        "Area equals width plus height: <<4+7=11>>. The area is 11.",
        "28",
    ),
    (
        "smoke_0004",
        "If 3 shirts cost $45 total, what is the cost of one shirt?",
        "Divide total by count: <<45/3=15>>. Each shirt costs $15.",
        "Divide total by count: <<45/3=12>>. Each shirt costs $12.",
        "15",
    ),
    (
        "smoke_0005",
        "Sara reads 25 pages a day for 6 days. How many pages did she read?",
        "Multiply: <<25*6=150>>. Sara read 150 pages.",
        "Multiply: <<25*6=130>>. Sara read 130 pages.",
        "150",
    ),
]


def _build_record(
    problem_id: str,
    problem: str,
    chosen: str,
    rejected: str,
    answer: str,
    median_length: int,
) -> dict:
    score = score_trace(chosen, is_correct=True, median_length=median_length)
    prov = build_provenance(
        problem_id=problem_id,
        chosen_text=chosen,
        rejected_text=rejected,
        source_dataset="smoke",
        source_split="synthetic",
        generator_backend="smoke",
        generator_model="canned-reference-traces",
        generation_temperature=0.0,
        generation_seed=0,
        verifier_verdict_chosen="accept",
        verifier_verdict_rejected="reject",
        verifier_version="v1",
        quality_score=score,
    )
    return {
        "pair": {
            "problem_id": problem_id,
            "problem": problem,
            "prompt": problem,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_answer": answer,
            "rejected_answer": "",
            "expected_answer": answer,
        },
        "provenance": prov.to_dict(),
        "quality_score": score,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "samples" / "synthetic_smoke",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    # Clear any prior smoke shards so the run is reproducible.
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("pairs_*.jsonl"):
            p.unlink()
    ckpt = PairCheckpointer(ckpt_dir, shard_size=3)
    stats = RunStats()

    lengths = [len(c) for _, _, c, _, _ in _BANK]
    median_length = int(statistics.median(lengths))

    for problem_id, problem, chosen, rejected, answer in _BANK:
        stats.record_problem_attempt()
        stats.record_generation(backend="smoke", accepted=True)
        stats.record_generation(backend="smoke", accepted=False)
        record = _build_record(problem_id, problem, chosen, rejected, answer, median_length)
        stats.record_pair_candidate()
        if ckpt.add(record):
            stats.record_unique_pair(record["quality_score"])
    ckpt.flush()

    # Flat artifact for inspection.
    flat = out / "dpo_pairs_v1.jsonl"
    with open(flat, "w", encoding="utf-8") as fh:
        for rec in ckpt.iter_all():
            fh.write(json.dumps(rec) + "\n")

    stats.write(out / "stats.json")

    readme = out / "README.md"
    readme.write_text(
        "# synthetic_smoke\n\n"
        "This directory is **not a real training run**. It is a tiny, CPU-only\n"
        "smoke artifact set used to verify the v1 provenance envelope, the\n"
        "quality-score components, and the resumable checkpointer end to end\n"
        "without invoking the LLM.\n\n"
        "Files:\n\n"
        "- `dpo_pairs_v1.jsonl` — flat list of v1 pair records.\n"
        "- `checkpoints/pairs_*.jsonl` — sharded checkpoint files.\n"
        "- `stats.json` — run-level counters and quality-score percentiles.\n\n"
        "To produce real data, see the `Synthetic Data Builder` section of\n"
        "the top-level README.\n"
    )

    logger.info("Wrote %d pairs to %s", ckpt.state.total_pairs, flat)
    logger.info("Stats: %s", stats.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
