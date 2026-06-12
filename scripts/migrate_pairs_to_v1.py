#!/usr/bin/env python3
"""
Rewrite a legacy `dpo_pairs.jsonl` (flat {prompt, chosen, rejected}) into
the v1 schema with provenance + quality_score.

Idempotent: lines already in v1 shape (a `provenance` block with a
matching `schema_version`) are passed through unchanged.

Usage:
    python scripts/migrate_pairs_to_v1.py data/dpo_pairs.jsonl
    python scripts/migrate_pairs_to_v1.py data/dpo_pairs.jsonl --output data/dpo_pairs_v1.jsonl

If --output is omitted the file is rewritten in place via a temp file
(atomic rename), and a `.bak` copy of the original is left next to it on
first migration.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

# Allow direct execution (`python scripts/...`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_generator.provenance import (  # noqa: E402
    PROVENANCE_SCHEMA_VERSION,
    build_provenance,
)


logger = logging.getLogger("migrate_pairs_to_v1")


def is_v1(record: Dict[str, Any]) -> bool:
    prov = record.get("provenance")
    if not isinstance(prov, dict):
        return False
    return prov.get("schema_version") == PROVENANCE_SCHEMA_VERSION


def to_v1(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map a legacy {prompt,chosen,rejected} record to the v1 envelope.

    Unknown provenance fields fall back to `legacy` or `None`. The
    quality_score is left null since we have no verifier verdict on
    archival data.
    """
    prompt = record.get("prompt") or record.get("problem") or ""
    chosen = record.get("chosen", "")
    rejected = record.get("rejected", "")
    problem_id = record.get("problem_id") or _stable_problem_id(prompt)

    provenance = build_provenance(
        problem_id=problem_id,
        chosen_text=chosen,
        rejected_text=rejected,
        source_dataset=record.get("source_dataset", "legacy"),
        source_split=record.get("source_split", "unknown"),
        generator_backend=record.get("generator_backend", "legacy"),
        generator_model=record.get("generator_model", "unknown"),
        generation_temperature=float(record.get("generation_temperature", 0.0)),
        generation_seed=record.get("generation_seed"),
        verifier_verdict_chosen=record.get("verifier_verdict_chosen", "unknown"),
        verifier_verdict_rejected=record.get("verifier_verdict_rejected", "unknown"),
        verifier_version=record.get("verifier_version", "legacy"),
        quality_score=None,
    )
    return {
        "pair": {
            "problem_id": problem_id,
            "problem": prompt,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_answer": record.get("chosen_answer"),
            "rejected_answer": record.get("rejected_answer"),
            "expected_answer": record.get("expected_answer"),
        },
        "provenance": provenance.to_dict(),
        "quality_score": None,
    }


def _stable_problem_id(prompt: str) -> str:
    import hashlib

    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"legacy_{h}"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", n, path, exc)


def migrate(records: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for rec in records:
        if is_v1(rec):
            yield rec
        else:
            yield to_v1(rec)


def migrate_file(input_path: Path, output_path: Path | None = None, make_backup: bool = True) -> Dict[str, int]:
    in_place = output_path is None
    counts = {"total": 0, "migrated": 0, "already_v1": 0}

    if in_place:
        # Write to a sibling temp file then rename.
        fd, tmp_name = tempfile.mkstemp(
            prefix=input_path.name + ".",
            suffix=".tmp",
            dir=str(input_path.parent),
        )
        import os

        os.close(fd)
        tmp_path = Path(tmp_name)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path

    with open(tmp_path, "w", encoding="utf-8") as out_fh:
        for rec in _iter_jsonl(input_path):
            counts["total"] += 1
            if is_v1(rec):
                counts["already_v1"] += 1
                out_fh.write(json.dumps(rec) + "\n")
            else:
                counts["migrated"] += 1
                out_fh.write(json.dumps(to_v1(rec)) + "\n")

    if in_place:
        if make_backup and counts["migrated"] > 0:
            bak = input_path.with_suffix(input_path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(input_path, bak)
                logger.info("Wrote backup %s", bak)
        tmp_path.replace(input_path)

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input dpo_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="output path (default: in place)")
    parser.add_argument("--no-backup", action="store_true", help="skip .bak file when migrating in place")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.input.exists():
        logger.error("input %s does not exist", args.input)
        return 1

    counts = migrate_file(args.input, args.output, make_backup=not args.no_backup)
    logger.info(
        "Done. total=%d migrated=%d already_v1=%d",
        counts["total"],
        counts["migrated"],
        counts["already_v1"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
