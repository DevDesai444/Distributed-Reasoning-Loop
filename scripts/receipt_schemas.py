#!/usr/bin/env python3
"""
Versioned receipt envelopes for synthetic / agreement / robustness runs.

A "receipt" is a thin normalization layer on top of whatever the existing
M1 CLIs already emit. The point is that every later consumer (the README
generator in R3, future receipt-to-receipt diffs, plot scripts, etc.)
reads one canonical JSON per run kind instead of poking around in the
nested output directory of a particular CLI version.

Dataclasses (no Pydantic) deliberately: the schemas are simple,
serialization is JSON, and Kaggle's preinstalled environment is fussy
about typing-extensions / pydantic v1-vs-v2 conflicts. Manual
`validate()` is enough.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


RECEIPT_SCHEMA_VERSION = "v1"


# -- helpers ----------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    """Return the current git commit, or 'unknown' if git isn't reachable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _require(d: Dict[str, Any], key: str, kind: str) -> Any:
    if key not in d:
        raise ValueError(f"{kind} receipt missing required field {key!r}")
    return d[key]


# -- agreement --------------------------------------------------------------


@dataclass
class AgreementPair:
    pair_name: str
    cohens_kappa: Optional[float]
    kappa_ci_95: List[Optional[float]]
    agreement_rate: Optional[float]
    confusion: Dict[str, int]
    per_difficulty: Dict[str, Dict[str, Any]]


@dataclass
class ShortcutBucket:
    total_verifier_accepted: int
    judge_flagged: int
    shortcut_rate: float
    by_reason: Dict[str, int]
    examples_path: str


@dataclass
class AgreementReceipt:
    schema_version: str
    kind: str
    run_id: str
    timestamp_utc: str
    git_commit: str
    model: str
    judge_model: str
    judge_prompt_version: str
    benchmark: str
    split: str
    n_problems: int
    n_traces: int
    rm_threshold: Optional[float]
    pairs: List[AgreementPair]
    shortcut_bucket: ShortcutBucket
    wall_clock_seconds: float
    hardware: Dict[str, Any]
    notes: Dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_run_artifacts(
        cls,
        run_dir: Path,
        *,
        wall_clock_seconds: float,
        hardware: Dict[str, Any],
        notes: Optional[Dict[str, Any]] = None,
    ) -> "AgreementReceipt":
        """Construct a receipt from the directory `scripts/agreement_eval.py`
        produced (or from the smoke artifact directory, which has the same
        shape minus the per-bin RM rows).
        """
        run_dir = Path(run_dir)
        # The full runner writes agreement_results.json; the smoke writes
        # agreement_smoke.json with the same field shape.
        candidates = [
            run_dir / "agreement_results.json",
            run_dir / "agreement_smoke.json",
        ]
        results_path = next((p for p in candidates if p.exists()), None)
        if results_path is None:
            raise FileNotFoundError(
                f"no agreement_results.json (or agreement_smoke.json) under {run_dir}"
            )
        results = _read_json(results_path)
        shortcuts_summary_path = run_dir / "shortcuts_summary.json"
        shortcuts_summary = (
            _read_json(shortcuts_summary_path)
            if shortcuts_summary_path.exists()
            else {
                "n_verifier_accepted": 0,
                "n_shortcuts": 0,
                "shortcut_rate": 0.0,
                "by_reason": {},
            }
        )

        pairs: List[AgreementPair] = []
        for pr in results.get("pair_reports", []):
            confusion = dict(pr.get("confusion", {}))
            per_bin = pr.get("per_bin", {}) or {}
            per_difficulty: Dict[str, Dict[str, Any]] = {}
            # Preserve whatever bucket names the benchmark used (GSM8K =
            # low/mid/high, MATH = level_1..5, HumanEval = "all").
            for bucket, row in per_bin.items():
                row = row or {}
                per_difficulty[bucket] = {
                    "kappa": row.get("kappa"),
                    "agreement": row.get("agreement"),
                    "n": int(row.get("n", 0)),
                }
            pairs.append(AgreementPair(
                pair_name=f"{pr.get('rater_a')}_vs_{pr.get('rater_b')}",
                cohens_kappa=pr.get("kappa"),
                kappa_ci_95=[pr.get("kappa_ci_low"), pr.get("kappa_ci_high")],
                agreement_rate=pr.get("agreement"),
                confusion=confusion,
                per_difficulty=per_difficulty,
            ))

        bucket = ShortcutBucket(
            total_verifier_accepted=int(shortcuts_summary.get("n_verifier_accepted", 0)),
            judge_flagged=int(shortcuts_summary.get("n_shortcuts", 0)),
            shortcut_rate=float(shortcuts_summary.get("shortcut_rate", 0.0) or 0.0),
            by_reason=dict(shortcuts_summary.get("by_reason", {})),
            examples_path=str((run_dir / "shortcuts.jsonl").relative_to(run_dir.parent)
                              if (run_dir / "shortcuts.jsonl").exists()
                              else "shortcuts.jsonl"),
        )

        metadata = results.get("metadata", {}) or {}
        return cls(
            schema_version=RECEIPT_SCHEMA_VERSION,
            kind="agreement",
            run_id=results.get("run_id", "agreement_unknown"),
            timestamp_utc=metadata.get("timestamp_utc") or _now_iso(),
            git_commit=_git_commit(),
            model=results.get("model", "unknown"),
            judge_model=results.get("judge_model", "unknown"),
            judge_prompt_version=results.get("judge_prompt_version", ""),
            benchmark=results.get("benchmark", "unknown"),
            split=metadata.get("split", "test"),
            n_problems=int(results.get("n_problems", 0)),
            n_traces=int(results.get("n_traces", 0)),
            rm_threshold=metadata.get("rm_threshold"),
            pairs=pairs,
            shortcut_bucket=bucket,
            wall_clock_seconds=float(wall_clock_seconds),
            hardware=dict(hardware),
            notes=dict(notes or {}),
        )

    # -- (de)serialization -------------------------------------------------

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgreementReceipt":
        pairs = [
            AgreementPair(**{k: p.get(k) for k in (
                "pair_name", "cohens_kappa", "kappa_ci_95",
                "agreement_rate", "confusion", "per_difficulty",
            )})
            for p in d.get("pairs", [])
        ]
        sb = d.get("shortcut_bucket", {})
        bucket = ShortcutBucket(
            total_verifier_accepted=int(sb.get("total_verifier_accepted", 0)),
            judge_flagged=int(sb.get("judge_flagged", 0)),
            shortcut_rate=float(sb.get("shortcut_rate", 0.0)),
            by_reason=dict(sb.get("by_reason", {})),
            examples_path=str(sb.get("examples_path", "")),
        )
        return cls(
            schema_version=str(_require(d, "schema_version", "agreement")),
            kind=str(_require(d, "kind", "agreement")),
            run_id=str(_require(d, "run_id", "agreement")),
            timestamp_utc=str(_require(d, "timestamp_utc", "agreement")),
            git_commit=str(d.get("git_commit", "unknown")),
            model=str(_require(d, "model", "agreement")),
            judge_model=str(_require(d, "judge_model", "agreement")),
            judge_prompt_version=str(d.get("judge_prompt_version", "")),
            benchmark=str(_require(d, "benchmark", "agreement")),
            split=str(d.get("split", "test")),
            n_problems=int(_require(d, "n_problems", "agreement")),
            n_traces=int(_require(d, "n_traces", "agreement")),
            rm_threshold=d.get("rm_threshold"),
            pairs=pairs,
            shortcut_bucket=bucket,
            wall_clock_seconds=float(d.get("wall_clock_seconds", 0.0)),
            hardware=dict(d.get("hardware", {})),
            notes=dict(d.get("notes", {})),
        )

    @classmethod
    def from_file(cls, path: Path) -> "AgreementReceipt":
        return cls.from_dict(_read_json(Path(path)))

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                f"agreement receipt schema version {self.schema_version!r} "
                f"!= expected {RECEIPT_SCHEMA_VERSION!r}"
            )
        if self.kind != "agreement":
            raise ValueError(f"agreement receipt has wrong kind: {self.kind!r}")
        if self.n_problems < 0 or self.n_traces < 0:
            raise ValueError("n_problems and n_traces must be non-negative")
        for pair in self.pairs:
            if not pair.pair_name:
                raise ValueError("agreement pair missing pair_name")
            if len(pair.kappa_ci_95) != 2:
                raise ValueError(
                    f"agreement pair {pair.pair_name} kappa_ci_95 must have 2 entries"
                )
        # shortcut bucket sanity
        if not 0.0 <= self.shortcut_bucket.shortcut_rate <= 1.0:
            raise ValueError(
                f"shortcut_rate out of [0,1]: {self.shortcut_bucket.shortcut_rate}"
            )


# -- robustness -------------------------------------------------------------


@dataclass
class PerturbationRow:
    name: str
    rewrites_label: bool
    applicable_rate: float
    matched_baseline_pass_at_1: float
    perturbed_pass_at_1: float
    retention: Optional[float]
    retention_ci_95: List[Optional[float]]


@dataclass
class RobustnessReceipt:
    schema_version: str
    kind: str
    run_id: str
    timestamp_utc: str
    git_commit: str
    model: str
    benchmark: str
    split: str
    n_problems: int
    n_samples_per_problem: int
    seed: int
    overall_robustness: Optional[float]
    per_perturbation: List[PerturbationRow]
    wall_clock_seconds: float
    hardware: Dict[str, Any]
    notes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_artifacts(
        cls,
        run_dir: Path,
        *,
        wall_clock_seconds: float,
        hardware: Dict[str, Any],
        notes: Optional[Dict[str, Any]] = None,
    ) -> "RobustnessReceipt":
        run_dir = Path(run_dir)
        results_path = run_dir / "robustness_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"no robustness_results.json under {run_dir}")
        results = _read_json(results_path)

        rows: List[PerturbationRow] = []
        for r in results.get("per_perturbation", []):
            ci = r.get("retention_ci", [None, None]) or [None, None]
            rows.append(PerturbationRow(
                name=str(r.get("name", "")),
                rewrites_label=bool(r.get("rewrites_label", False)),
                applicable_rate=float(r.get("applicability_rate", 0.0) or 0.0),
                matched_baseline_pass_at_1=float(
                    r.get("matched_baseline_pass_at_1", 0.0) or 0.0
                ),
                perturbed_pass_at_1=float(
                    r.get("perturbed_pass_at_1", 0.0) or 0.0
                ),
                retention=r.get("retention"),
                retention_ci_95=[ci[0] if len(ci) > 0 else None,
                                 ci[1] if len(ci) > 1 else None],
            ))

        metadata = results.get("metadata", {}) or {}
        manifest_path = run_dir / "run_manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}

        return cls(
            schema_version=RECEIPT_SCHEMA_VERSION,
            kind="robustness",
            run_id=results.get("run_id", manifest.get("run_id", "robustness_unknown")),
            timestamp_utc=metadata.get("timestamp_utc")
                          or manifest.get("timestamp_utc") or _now_iso(),
            git_commit=_git_commit(),
            model=results.get("model", manifest.get("model", "unknown")),
            benchmark=results.get("benchmark", manifest.get("benchmark", "gsm8k")),
            split=metadata.get("split") or manifest.get("split", "test"),
            n_problems=int(results.get("n_problems", manifest.get("n_problems", 0))),
            n_samples_per_problem=int(
                results.get("n_samples_per_problem",
                            manifest.get("n_samples_per_problem", 0))
            ),
            seed=int(results.get("seed", manifest.get("seed", 0))),
            overall_robustness=results.get("overall_robustness",
                                           manifest.get("overall_robustness")),
            per_perturbation=rows,
            wall_clock_seconds=float(wall_clock_seconds),
            hardware=dict(hardware),
            notes=dict(notes or {}),
        )

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RobustnessReceipt":
        rows = [
            PerturbationRow(**{k: r.get(k) for k in (
                "name", "rewrites_label", "applicable_rate",
                "matched_baseline_pass_at_1", "perturbed_pass_at_1",
                "retention", "retention_ci_95",
            )})
            for r in d.get("per_perturbation", [])
        ]
        return cls(
            schema_version=str(_require(d, "schema_version", "robustness")),
            kind=str(_require(d, "kind", "robustness")),
            run_id=str(_require(d, "run_id", "robustness")),
            timestamp_utc=str(_require(d, "timestamp_utc", "robustness")),
            git_commit=str(d.get("git_commit", "unknown")),
            model=str(_require(d, "model", "robustness")),
            benchmark=str(_require(d, "benchmark", "robustness")),
            split=str(d.get("split", "test")),
            n_problems=int(_require(d, "n_problems", "robustness")),
            n_samples_per_problem=int(d.get("n_samples_per_problem", 0)),
            seed=int(d.get("seed", 0)),
            overall_robustness=d.get("overall_robustness"),
            per_perturbation=rows,
            wall_clock_seconds=float(d.get("wall_clock_seconds", 0.0)),
            hardware=dict(d.get("hardware", {})),
            notes=dict(d.get("notes", {})),
        )

    @classmethod
    def from_file(cls, path: Path) -> "RobustnessReceipt":
        return cls.from_dict(_read_json(Path(path)))

    def validate(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                f"robustness receipt schema version {self.schema_version!r} "
                f"!= expected {RECEIPT_SCHEMA_VERSION!r}"
            )
        if self.kind != "robustness":
            raise ValueError(f"robustness receipt has wrong kind: {self.kind!r}")
        if self.n_problems < 0 or self.n_samples_per_problem < 0:
            raise ValueError("n_problems and n_samples_per_problem must be non-negative")
        for row in self.per_perturbation:
            if not row.name:
                raise ValueError("perturbation row missing name")
            if len(row.retention_ci_95) != 2:
                raise ValueError(
                    f"perturbation {row.name} retention_ci_95 must have 2 entries"
                )


# -- synthetic --------------------------------------------------------------


@dataclass
class SyntheticReceipt:
    schema_version: str
    kind: str
    run_id: str
    timestamp_utc: str
    git_commit: str
    model: str
    dataset: str
    split: str
    target_pairs: Optional[int]
    stats: Dict[str, Any]
    wall_clock_seconds: float
    hardware: Dict[str, Any]
    notes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_artifacts(
        cls,
        run_dir: Path,
        *,
        wall_clock_seconds: float,
        hardware: Dict[str, Any],
        model: str,
        dataset: str = "gsm8k",
        split: str = "train",
        target_pairs: Optional[int] = None,
        run_id: Optional[str] = None,
        notes: Optional[Dict[str, Any]] = None,
    ) -> "SyntheticReceipt":
        run_dir = Path(run_dir)
        stats_path = run_dir / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"no stats.json under {run_dir}")
        stats = _read_json(stats_path)
        # `pairs_*.jsonl` lives under checkpoints/, plus a flat
        # dpo_pairs_v1.jsonl in the full pipeline.  We don't need to read
        # them for the receipt — stats.json already aggregates.
        rid = run_id or datetime.now(timezone.utc).strftime("synthetic_%Y%m%dT%H%M%SZ")

        return cls(
            schema_version=RECEIPT_SCHEMA_VERSION,
            kind="synthetic",
            run_id=rid,
            timestamp_utc=stats.get("completed_at") or _now_iso(),
            git_commit=_git_commit(),
            model=model,
            dataset=dataset,
            split=split,
            target_pairs=target_pairs,
            stats={
                "total_problems_attempted": int(stats.get("total_problems_attempted", 0)),
                "total_generations": int(stats.get("total_generations", 0)),
                "verifier_accepts": int(stats.get("verifier_accepts", 0)),
                "verifier_rejects": int(stats.get("verifier_rejects", 0)),
                "verifier_accept_rate": float(stats.get("verifier_accept_rate", 0.0) or 0.0),
                "pair_candidates": int(stats.get("pair_candidates", 0)),
                "unique_pairs_after_dedup": int(stats.get("unique_pairs_after_dedup", 0)),
                "dedup_ratio": float(stats.get("dedup_ratio", 0.0) or 0.0),
                "per_backend": dict(stats.get("per_backend", {})),
                "quality_score_distribution": dict(
                    stats.get("quality_score_distribution", {})
                ),
            },
            wall_clock_seconds=float(wall_clock_seconds),
            hardware=dict(hardware),
            notes=dict(notes or {}),
        )

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyntheticReceipt":
        return cls(
            schema_version=str(_require(d, "schema_version", "synthetic")),
            kind=str(_require(d, "kind", "synthetic")),
            run_id=str(_require(d, "run_id", "synthetic")),
            timestamp_utc=str(_require(d, "timestamp_utc", "synthetic")),
            git_commit=str(d.get("git_commit", "unknown")),
            model=str(_require(d, "model", "synthetic")),
            dataset=str(d.get("dataset", "gsm8k")),
            split=str(d.get("split", "train")),
            target_pairs=d.get("target_pairs"),
            stats=dict(d.get("stats", {})),
            wall_clock_seconds=float(d.get("wall_clock_seconds", 0.0)),
            hardware=dict(d.get("hardware", {})),
            notes=dict(d.get("notes", {})),
        )

    @classmethod
    def from_file(cls, path: Path) -> "SyntheticReceipt":
        return cls.from_dict(_read_json(Path(path)))

    def validate(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                f"synthetic receipt schema version {self.schema_version!r} "
                f"!= expected {RECEIPT_SCHEMA_VERSION!r}"
            )
        if self.kind != "synthetic":
            raise ValueError(f"synthetic receipt has wrong kind: {self.kind!r}")
        required_stats = (
            "total_problems_attempted",
            "total_generations",
            "verifier_accepts",
            "unique_pairs_after_dedup",
        )
        for key in required_stats:
            if key not in self.stats:
                raise ValueError(f"synthetic receipt missing stats.{key}")


# -- registry / convenience -------------------------------------------------


RECEIPT_FILENAMES = {
    "synthetic": "synthetic_v1.json",
    "agreement": "agreement_v1.json",
    "robustness": "robustness_v1.json",
}


def receipt_path(output_dir: Path, kind: str) -> Path:
    output_dir = Path(output_dir)
    if kind not in RECEIPT_FILENAMES:
        raise KeyError(f"unknown receipt kind: {kind!r}")
    return output_dir / "receipts" / RECEIPT_FILENAMES[kind]


def load_receipt(path: Path):
    """Dispatch based on the `kind` field."""
    d = _read_json(Path(path))
    kind = d.get("kind")
    if kind == "agreement":
        return AgreementReceipt.from_dict(d)
    if kind == "robustness":
        return RobustnessReceipt.from_dict(d)
    if kind == "synthetic":
        return SyntheticReceipt.from_dict(d)
    raise ValueError(f"unknown receipt kind in {path}: {kind!r}")
