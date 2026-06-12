"""Schema-conformance tests for emitted JSONL and the migration helper."""

import json
from pathlib import Path

import pytest

from data_generator.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    build_provenance,
)
from scripts.migrate_pairs_to_v1 import is_v1, migrate_file, to_v1


REQUIRED_PROVENANCE_FIELDS = {
    "pair_id",
    "problem_id",
    "source_dataset",
    "source_split",
    "generator_backend",
    "generator_model",
    "generation_temperature",
    "generation_seed",
    "verifier_verdict_chosen",
    "verifier_verdict_rejected",
    "verifier_version",
    "dedup_hash_chosen",
    "dedup_hash_rejected",
    "quality_score",
    "generated_at",
    "schema_version",
}


def _assert_v1_envelope(record):
    assert set(record.keys()) >= {"pair", "provenance", "quality_score"}
    assert isinstance(record["pair"], dict)
    assert isinstance(record["provenance"], dict)
    missing = REQUIRED_PROVENANCE_FIELDS - record["provenance"].keys()
    assert not missing, f"missing provenance fields: {missing}"
    assert record["provenance"]["schema_version"] == PROVENANCE_SCHEMA_VERSION


def test_build_provenance_conforms(tmp_path):
    prov = build_provenance(
        problem_id="p1",
        chosen_text="c",
        rejected_text="r",
        source_dataset="gsm8k",
        source_split="train",
        generator_backend="vllm",
        generator_model="x",
        generation_temperature=0.7,
        generation_seed=42,
        verifier_verdict_chosen="accept",
        verifier_verdict_rejected="reject",
        verifier_version="v1",
        quality_score=0.9,
    ).to_dict()
    assert REQUIRED_PROVENANCE_FIELDS.issubset(prov.keys())


def test_migration_produces_v1_envelope(tmp_path: Path):
    legacy = tmp_path / "dpo_pairs.jsonl"
    with open(legacy, "w") as fh:
        for i in range(3):
            fh.write(json.dumps({
                "prompt": f"problem {i}",
                "chosen": f"good {i}",
                "rejected": f"bad {i}",
            }) + "\n")

    counts = migrate_file(legacy)
    assert counts["total"] == 3
    assert counts["migrated"] == 3
    assert counts["already_v1"] == 0

    with open(legacy) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(records) == 3
    for rec in records:
        _assert_v1_envelope(rec)
        assert rec["quality_score"] is None  # legacy has no verifier verdict
        assert rec["provenance"]["generator_backend"] == "legacy"


def test_migration_is_idempotent(tmp_path: Path):
    legacy = tmp_path / "dpo_pairs.jsonl"
    with open(legacy, "w") as fh:
        for i in range(2):
            fh.write(json.dumps({
                "prompt": f"problem {i}",
                "chosen": f"good {i}",
                "rejected": f"bad {i}",
            }) + "\n")

    # First migration: rewrites both.
    first = migrate_file(legacy)
    assert first["migrated"] == 2

    # Capture content for byte-level equality on second run.
    with open(legacy) as fh:
        contents = fh.read()

    # Second migration: nothing to do.
    second = migrate_file(legacy, make_backup=False)
    assert second["already_v1"] == 2
    assert second["migrated"] == 0
    with open(legacy) as fh:
        assert fh.read() == contents


def test_migration_preserves_v1_lines_unchanged(tmp_path: Path):
    out = tmp_path / "dpo_pairs.jsonl"
    # One legacy, one already-v1.
    v1_record = to_v1({"prompt": "x", "chosen": "c", "rejected": "r"})
    with open(out, "w") as fh:
        fh.write(json.dumps({"prompt": "y", "chosen": "c2", "rejected": "r2"}) + "\n")
        fh.write(json.dumps(v1_record) + "\n")

    counts = migrate_file(out, make_backup=False)
    assert counts["total"] == 2
    assert counts["migrated"] == 1
    assert counts["already_v1"] == 1

    with open(out) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert is_v1(records[0])
    assert is_v1(records[1])


def test_migration_with_explicit_output(tmp_path: Path):
    src = tmp_path / "src.jsonl"
    dst = tmp_path / "out" / "dst.jsonl"
    with open(src, "w") as fh:
        fh.write(json.dumps({"prompt": "x", "chosen": "c", "rejected": "r"}) + "\n")

    counts = migrate_file(src, output_path=dst)
    assert counts["migrated"] == 1
    assert dst.exists()
    # Source untouched.
    with open(src) as fh:
        assert json.loads(fh.read().strip()) == {
            "prompt": "x", "chosen": "c", "rejected": "r",
        }
