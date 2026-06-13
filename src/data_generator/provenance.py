"""
Provenance metadata for synthetic preference pairs.

Every emitted pair carries a `ProvenanceMetadata` blob so that downstream
consumers can reproduce, audit, or slice the dataset. The schema version is
bumped whenever the field set changes so old artifacts can be detected and
migrated.

Hash functions here are deliberately deterministic: same inputs always yield
the same id. This is what makes dedup correct on resume.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


PROVENANCE_SCHEMA_VERSION = "1"

# Whitespace collapser for dedup_hash. We strip every run of whitespace down to
# a single space and lowercase, so that two traces differing only in indentation
# or trailing newlines hash to the same value.
_WS_RE = re.compile(r"\s+")


def _canonical(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip().lower()


def dedup_hash(text: str) -> str:
    """sha256 of whitespace-normalized text.

    Used to identify identical generations across runs. Collisions on
    sha256 are astronomically unlikely; equality of hashes means the
    generations are the same modulo whitespace.
    """
    return hashlib.sha256(_canonical(text).encode("utf-8")).hexdigest()


def pair_id(problem_id: str, chosen: str, rejected: str) -> str:
    """Stable identifier for a (problem, chosen, rejected) triple.

    Computed over the canonicalized chosen/rejected text so that surface-level
    whitespace differences do not produce a new id.
    """
    payload = "|".join([problem_id or "", _canonical(chosen), _canonical(rejected)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ProvenanceMetadata:
    """Per-pair metadata block. See PHASE 3 plan for field semantics."""

    pair_id: str
    problem_id: str
    source_dataset: str
    source_split: str
    generator_backend: str
    generator_model: str
    generation_temperature: float
    generation_seed: Optional[int]
    verifier_verdict_chosen: str
    verifier_verdict_rejected: str
    verifier_version: str
    dedup_hash_chosen: str
    dedup_hash_rejected: str
    quality_score: Optional[float] = None
    generated_at: str = field(default_factory=utc_now_iso)
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceMetadata":
        # Tolerate missing optional fields by filling defaults
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        cleaned = {k: v for k, v in data.items() if k in known}
        return cls(**cleaned)


def build_provenance(
    *,
    problem_id: str,
    chosen_text: str,
    rejected_text: str,
    source_dataset: str,
    source_split: str,
    generator_backend: str,
    generator_model: str,
    generation_temperature: float,
    generation_seed: Optional[int],
    verifier_verdict_chosen: str,
    verifier_verdict_rejected: str,
    verifier_version: str,
    quality_score: Optional[float] = None,
    generated_at: Optional[str] = None,
) -> ProvenanceMetadata:
    """Convenience constructor that derives the hash fields from text."""
    return ProvenanceMetadata(
        pair_id=pair_id(problem_id, chosen_text, rejected_text),
        problem_id=problem_id,
        source_dataset=source_dataset,
        source_split=source_split,
        generator_backend=generator_backend,
        generator_model=generator_model,
        generation_temperature=generation_temperature,
        generation_seed=generation_seed,
        verifier_verdict_chosen=verifier_verdict_chosen,
        verifier_verdict_rejected=verifier_verdict_rejected,
        verifier_version=verifier_version,
        dedup_hash_chosen=dedup_hash(chosen_text),
        dedup_hash_rejected=dedup_hash(rejected_text),
        quality_score=quality_score,
        generated_at=generated_at or utc_now_iso(),
    )


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceMetadata",
    "build_provenance",
    "dedup_hash",
    "pair_id",
    "utc_now_iso",
]
