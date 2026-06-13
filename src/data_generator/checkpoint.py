"""
Resumable checkpoints for the synthetic data pipeline.

We write JSONL shards of size N (default 1000) named `pairs_<seq>.jsonl`
where seq is the rolling pair count at the end of the shard. The shards
are append-safe per-line: a crash leaves a valid prefix.

On resume we read every shard, populate the seen-set with `pair_id`s and
the seen-problems set with `problem_id`s, and skip those when iterating.

Idempotence: writing a pair whose pair_id is already in the seen-set is a
no-op. This makes re-runs of the same input a no-op too.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    seen_pair_ids: Set[str] = field(default_factory=set)
    seen_problem_ids: Set[str] = field(default_factory=set)
    total_pairs: int = 0
    shard_index: int = 0


class PairCheckpointer:
    """Writes pairs to size-N JSONL shards under `checkpoint_dir`."""

    def __init__(
        self,
        checkpoint_dir: Path,
        shard_size: int = 1000,
        prefix: str = "pairs_",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1, shard_size)
        self.prefix = prefix
        self.state = CheckpointState()
        self._buffer: List[Dict[str, Any]] = []

    # --- restore ---------------------------------------------------------

    def restore(self) -> CheckpointState:
        """Rebuild the seen-sets from existing shards.

        Idempotent. Safe to call when nothing exists yet.
        """
        shards = sorted(
            self.checkpoint_dir.glob(f"{self.prefix}*.jsonl"),
            key=_shard_sort_key,
        )
        for shard in shards:
            for record in _iter_jsonl(shard):
                prov = record.get("provenance", {}) or {}
                pid = prov.get("pair_id") or record.get("pair_id")
                problem_id = prov.get("problem_id") or record.get("problem_id")
                if pid:
                    self.state.seen_pair_ids.add(pid)
                if problem_id:
                    self.state.seen_problem_ids.add(problem_id)
                self.state.total_pairs += 1
        # Next shard index continues after the last full shard.
        self.state.shard_index = self.state.total_pairs // self.shard_size
        logger.info(
            "Restored checkpoint: %d shards, %d pairs, %d problems seen",
            len(shards),
            self.state.total_pairs,
            len(self.state.seen_problem_ids),
        )
        return self.state

    # --- writing ---------------------------------------------------------

    def add(self, record: Dict[str, Any]) -> bool:
        """Append a record. Returns True if accepted, False if duplicate.

        A record is a dict containing at minimum a `provenance` block with
        `pair_id` and `problem_id`.
        """
        prov = record.get("provenance", {}) or {}
        pid = prov.get("pair_id")
        if not pid:
            raise ValueError("record is missing provenance.pair_id")
        if pid in self.state.seen_pair_ids:
            return False
        self.state.seen_pair_ids.add(pid)
        problem_id = prov.get("problem_id")
        if problem_id:
            self.state.seen_problem_ids.add(problem_id)
        self._buffer.append(record)
        self.state.total_pairs += 1
        if len(self._buffer) >= self.shard_size:
            self._flush()
        return True

    def flush(self) -> None:
        """Force-flush any pending records to a final (partial) shard."""
        if self._buffer:
            self._flush()

    def _flush(self) -> None:
        self.state.shard_index += 1
        path = self.checkpoint_dir / f"{self.prefix}{self.state.total_pairs}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for record in self._buffer:
                fh.write(json.dumps(record) + "\n")
        logger.info("Wrote checkpoint shard %s (%d pairs)", path.name, len(self._buffer))
        self._buffer.clear()

    # --- iteration -------------------------------------------------------

    def iter_all(self) -> Iterator[Dict[str, Any]]:
        """Yield every record from every shard in order."""
        for shard in sorted(
            self.checkpoint_dir.glob(f"{self.prefix}*.jsonl"),
            key=_shard_sort_key,
        ):
            yield from _iter_jsonl(shard)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line in %s: %s", path, exc)


def _shard_sort_key(path: Path) -> Tuple[int, str]:
    # Sort by trailing integer if possible, otherwise by name.
    stem = path.stem
    try:
        n = int(stem.split("_")[-1])
    except ValueError:
        n = -1
    return (n, stem)


__all__ = ["CheckpointState", "PairCheckpointer"]
