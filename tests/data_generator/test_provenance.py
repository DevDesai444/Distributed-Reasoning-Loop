"""Tests for provenance utilities: pair_id determinism, dedup_hash properties."""

from data_generator.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    ProvenanceMetadata,
    build_provenance,
    dedup_hash,
    pair_id,
)


def test_pair_id_deterministic():
    a = pair_id("p1", "chosen text", "rejected text")
    b = pair_id("p1", "chosen text", "rejected text")
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_pair_id_depends_on_inputs():
    base = pair_id("p1", "chosen", "rejected")
    assert pair_id("p2", "chosen", "rejected") != base
    assert pair_id("p1", "different", "rejected") != base
    assert pair_id("p1", "chosen", "different") != base


def test_pair_id_canonicalizes_whitespace():
    # Two strings differing only in indentation/case should yield same id
    a = pair_id("p1", "Hello World", "X")
    b = pair_id("p1", "  hello   world\n", "x")
    assert a == b


def test_dedup_hash_canonical_whitespace():
    h1 = dedup_hash("step 1\nstep 2\n")
    h2 = dedup_hash("step 1 step 2")
    h3 = dedup_hash("Step 1\tStep 2")
    assert h1 == h2 == h3


def test_dedup_hash_distinguishes_content():
    assert dedup_hash("alpha") != dedup_hash("beta")


def test_build_provenance_populates_fields():
    prov = build_provenance(
        problem_id="gsm8k_train_0001",
        chosen_text="48 + 24 = 72",
        rejected_text="48 + 24 = 80",
        source_dataset="gsm8k",
        source_split="train",
        generator_backend="vllm",
        generator_model="Qwen/Qwen2.5-1.5B-Instruct",
        generation_temperature=0.7,
        generation_seed=42,
        verifier_verdict_chosen="accept",
        verifier_verdict_rejected="reject",
        verifier_version="v1",
        quality_score=0.83,
    )
    d = prov.to_dict()
    assert d["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert d["pair_id"] == pair_id("gsm8k_train_0001", "48 + 24 = 72", "48 + 24 = 80")
    assert d["dedup_hash_chosen"] == dedup_hash("48 + 24 = 72")
    assert d["dedup_hash_rejected"] == dedup_hash("48 + 24 = 80")
    assert d["quality_score"] == 0.83
    assert d["verifier_verdict_chosen"] == "accept"
    assert d["verifier_verdict_rejected"] == "reject"
    assert "generated_at" in d and "T" in d["generated_at"]


def test_provenance_from_dict_roundtrip():
    prov = build_provenance(
        problem_id="p1",
        chosen_text="a",
        rejected_text="b",
        source_dataset="gsm8k",
        source_split="train",
        generator_backend="vllm",
        generator_model="x",
        generation_temperature=0.7,
        generation_seed=None,
        verifier_verdict_chosen="accept",
        verifier_verdict_rejected="reject",
        verifier_version="v1",
    )
    again = ProvenanceMetadata.from_dict(prov.to_dict())
    assert again.to_dict() == prov.to_dict()


def test_provenance_tolerates_unknown_fields():
    payload = build_provenance(
        problem_id="p1",
        chosen_text="a",
        rejected_text="b",
        source_dataset="gsm8k",
        source_split="train",
        generator_backend="vllm",
        generator_model="x",
        generation_temperature=0.7,
        generation_seed=None,
        verifier_verdict_chosen="accept",
        verifier_verdict_rejected="reject",
        verifier_version="v1",
    ).to_dict()
    payload["future_field"] = "ignored"
    prov = ProvenanceMetadata.from_dict(payload)
    assert prov.problem_id == "p1"
