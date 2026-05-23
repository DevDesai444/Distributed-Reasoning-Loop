"""
Structured run artifact management for pipeline executions.

This module makes runs explicit, reproducible objects instead of a loose set of
files written into a shared output directory.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_value(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


@dataclass
class ArtifactRecord:
    stage: str
    name: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    pipeline: str
    dataset: str
    training_method: str
    root_output_dir: str
    run_dir: str
    cwd: str
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    status: str = "initialized"
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(record) for record in self.artifacts]
        return payload


class RunArtifacts:
    """Manage per-run directories plus a machine-readable run manifest."""

    def __init__(
        self,
        *,
        root_output_dir: str,
        dataset: str,
        training_method: str,
        pipeline: str = "distributed_reasoning_loop",
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        nested: bool = True,
    ):
        root = Path(root_output_dir)
        self.root_output_dir = root.resolve()
        normalized_name = run_name.strip().replace(" ", "_") if run_name else None
        run_id = normalized_name or f"{_utc_timestamp()}-{dataset}-{training_method}"
        if nested:
            runs_root = root / "runs"
            runs_root.mkdir(parents=True, exist_ok=True)
            self.run_dir = runs_root / run_id
            self.runs_root = runs_root
        else:
            root.mkdir(parents=True, exist_ok=True)
            self.run_dir = root
            self.runs_root = root
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            pipeline=pipeline,
            dataset=dataset,
            training_method=training_method,
            root_output_dir=str(root.resolve()),
            run_dir=str(self.run_dir.resolve()),
            cwd=os.getcwd(),
            git_commit=_git_value("rev-parse", "HEAD"),
            git_branch=_git_value("rev-parse", "--abbrev-ref", "HEAD"),
            config=config or {},
        )
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.summary_path = self.run_dir / "run_summary.json"
        self.write_manifest()
        self._write_latest_pointer(root)

    def _write_latest_pointer(self, root_output_dir: Path) -> None:
        pointer = {
            "run_id": self.manifest.run_id,
            "run_dir": str(self.run_dir.resolve()),
        }
        with open(root_output_dir / "latest_run.json", "w") as f:
            json.dump(pointer, f, indent=2)

    def write_manifest(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)

    def stage_dir(self, name: str) -> Path:
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_artifact(
        self,
        *,
        stage: str,
        name: str,
        path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        artifact_path = str(path)
        path_obj = Path(artifact_path)
        if path_obj.exists() or artifact_path.startswith((".", "/")):
            artifact_path = str(path_obj.resolve())
        self.manifest.artifacts.append(
            ArtifactRecord(
                stage=stage,
                name=name,
                path=artifact_path,
                metadata=metadata or {},
            )
        )
        self.write_manifest()

    def record_metric(self, name: str, value: Any) -> None:
        self.manifest.metrics[name] = value
        self.write_manifest()

    def add_note(self, message: str) -> None:
        self.manifest.notes.append(message)
        self.write_manifest()

    def finalize(self, status: str, summary: Optional[Dict[str, Any]] = None) -> None:
        self.manifest.status = status
        self.write_manifest()
        if summary is not None:
            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2)

    def _find_previous_summary_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        for entry in self.runs_root.iterdir():
            if not entry.is_dir() or entry.resolve() == self.run_dir.resolve():
                continue
            summary_path = entry / "run_summary.json"
            if summary_path.exists():
                candidates.append(summary_path)
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]

    def load_previous_summary(self) -> Optional[Dict[str, Any]]:
        summary_path = self._find_previous_summary_path()
        if summary_path is None:
            return None
        with open(summary_path) as handle:
            return json.load(handle)

    def _flatten_numeric_summary(
        self,
        payload: Any,
        prefix: str = "",
        output: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if output is None:
            output = {}

        if isinstance(payload, dict):
            for key, value in payload.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                self._flatten_numeric_summary(value, child_prefix, output)
            return output

        if isinstance(payload, (int, float)) and not isinstance(payload, bool):
            output[prefix] = float(payload)
        return output

    def compare_summary_to_previous(self, summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        previous_summary = self.load_previous_summary()
        if previous_summary is None:
            return None

        current_flat = self._flatten_numeric_summary(summary)
        previous_flat = self._flatten_numeric_summary(previous_summary)
        deltas = {}
        for key, current_value in current_flat.items():
            if key not in previous_flat:
                continue
            deltas[key] = {
                "previous": previous_flat[key],
                "current": current_value,
                "delta": current_value - previous_flat[key],
            }

        return {
            "previous_run_summary": str(self._find_previous_summary_path()),
            "metrics": deltas,
        }
