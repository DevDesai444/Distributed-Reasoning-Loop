"""Resumable checkpoint round-trip tests.

Validates:
- Pairs round-trip through shard files
- Resuming rebuilds the seen-set
- Duplicate pair_ids are rejected (idempotent on rerun)
- Sharding boundary is honored
"""

import json
from pathlib import Path

import pytest

from data_generator.checkpoint import PairCheckpointer
from data_generator.provenance import build_provenance


def _make_record(problem_id: str, chosen: str, rejected: str) -> dict:
    prov = build_provenance(
        problem_id=problem_id,
        chosen_text=chosen,
        rejected_text=rejected,
        source_dataset="gsm8k",
        source_split="train",
        generator_backend="vllm",
        generator_model="x",
        generation_temperature=0.7,
        generation_seed=42,
        verifier_verdict_chosen="accept",
        verifier_verdict_rejected="reject",
        verifier_version="v1",
        quality_score=0.5,
    )
    return {
        "pair": {"problem_id": problem_id, "chosen": chosen, "rejected": rejected},
        "provenance": prov.to_dict(),
        "quality_score": 0.5,
    }


def test_checkpoint_writes_shards(tmp_path: Path):
    ck = PairCheckpointer(tmp_path, shard_size=3)
    for i in range(7):
        ck.add(_make_record(f"p{i}", f"c{i}", f"r{i}"))
    ck.flush()

    shards = sorted(tmp_path.glob("pairs_*.jsonl"))
    # 3 + 3 + 1
    assert len(shards) == 3
    counts = []
    for s in shards:
        with open(s) as fh:
            counts.append(sum(1 for _ in fh))
    assert counts == [3, 3, 1]


def test_checkpoint_resume_rebuilds_seen_set(tmp_path: Path):
    ck = PairCheckpointer(tmp_path, shard_size=10)
    for i in range(5):
        ck.add(_make_record(f"p{i}", f"c{i}", f"r{i}"))
    ck.flush()

    # New instance: restore from disk.
    ck2 = PairCheckpointer(tmp_path, shard_size=10)
    state = ck2.restore()
    assert state.total_pairs == 5
    assert len(state.seen_pair_ids) == 5
    assert state.seen_problem_ids == {f"p{i}" for i in range(5)}


def test_duplicate_pair_id_is_rejected(tmp_path: Path):
    ck = PairCheckpointer(tmp_path, shard_size=10)
    rec = _make_record("p1", "c1", "r1")
    assert ck.add(rec) is True
    # Adding the same record again must be a no-op.
    assert ck.add(rec) is False
    ck.flush()
    # Only one shard, one line.
    shards = list(tmp_path.glob("pairs_*.jsonl"))
    assert len(shards) == 1
    with open(shards[0]) as fh:
        assert sum(1 for _ in fh) == 1


def test_resume_then_add_skips_duplicates(tmp_path: Path):
    # First run: 3 pairs.
    ck = PairCheckpointer(tmp_path, shard_size=10)
    for i in range(3):
        ck.add(_make_record(f"p{i}", f"c{i}", f"r{i}"))
    ck.flush()

    # Second run: re-add same 3 plus 2 new.
    ck2 = PairCheckpointer(tmp_path, shard_size=10)
    ck2.restore()
    new_count = 0
    for i in range(5):
        if ck2.add(_make_record(f"p{i}", f"c{i}", f"r{i}")):
            new_count += 1
    ck2.flush()

    assert new_count == 2  # 3 dupes rejected, 2 new accepted
    all_records = list(ck2.iter_all())
    pair_ids = {r["provenance"]["pair_id"] for r in all_records}
    assert len(pair_ids) == 5


def test_iter_all_yields_every_record(tmp_path: Path):
    ck = PairCheckpointer(tmp_path, shard_size=2)
    expected_ids = []
    for i in range(5):
        rec = _make_record(f"p{i}", f"c{i}", f"r{i}")
        expected_ids.append(rec["provenance"]["pair_id"])
        ck.add(rec)
    ck.flush()
    got = [r["provenance"]["pair_id"] for r in ck.iter_all()]
    assert got == expected_ids


def test_malformed_line_skipped_on_restore(tmp_path: Path):
    # Pre-seed a shard with one good line and one malformed line.
    shard = tmp_path / "pairs_2.jsonl"
    good = _make_record("p_good", "c", "r")
    with open(shard, "w") as fh:
        fh.write(json.dumps(good) + "\n")
        fh.write("not-json\n")

    ck = PairCheckpointer(tmp_path, shard_size=10)
    state = ck.restore()
    # The good record is counted; the malformed line is dropped.
    assert state.total_pairs == 1
    assert len(state.seen_pair_ids) == 1
