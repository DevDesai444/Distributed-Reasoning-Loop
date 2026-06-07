"""
Streaming tokenized pre-training data with deterministic sharding.

The dataset emits ``(input_ids, labels)`` pairs that are pre-packed to the
configured sequence length. Each rank reads only its own shard so the
``IterableDataset`` is FSDP-safe.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class PretrainingDataConfig:
    data_dir: str = "./data/pretraining"
    file_glob: str = "shard-*.jsonl"
    text_field: str = "text"
    seq_len: int = 2048
    pad_id: int = 0


class StreamingTokenizedDataset(IterableDataset):
    """Pre-packed pre-training shards consumed in deterministic, rank-aware order.

    The dataset expects tokenized JSONL shards where each line is either a list
    of token IDs or a JSON object exposing ``tokens``. Tokens are concatenated
    and chunked to ``seq_len + 1`` so labels can be the next-token shift.
    """

    def __init__(
        self,
        config: PretrainingDataConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = max(1, world_size)
        pattern = os.path.join(config.data_dir, config.file_glob)
        files = sorted(glob.glob(pattern))
        if not files:
            logger.info(
                "No pretraining shards matched %s; the dataset will yield nothing.",
                pattern,
            )
        self.files: List[str] = files

    def _own_shards(self) -> List[str]:
        return [f for i, f in enumerate(self.files) if i % self.world_size == self.rank]

    def __iter__(self) -> Iterator[dict]:
        buffer: List[int] = []
        block = self.config.seq_len + 1
        for path in self._own_shards():
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, list):
                        tokens = payload
                    elif isinstance(payload, dict):
                        tokens = payload.get("tokens", [])
                    else:
                        continue
                    buffer.extend(int(t) for t in tokens)
                    while len(buffer) >= block:
                        chunk = buffer[:block]
                        buffer = buffer[block:]
                        yield {
                            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                            "labels": torch.tensor(chunk[1:], dtype=torch.long),
                        }


def build_pretraining_loader(
    config: PretrainingDataConfig,
    *,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 2,
) -> DataLoader:
    dataset = StreamingTokenizedDataset(config, rank=rank, world_size=world_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def synthetic_token_shards(
    output_dir: str,
    *,
    num_shards: int = 4,
    tokens_per_shard: int = 16_384,
    vocab_size: int = 4096,
    seed: int = 0,
) -> List[str]:
    """Write deterministic synthetic shards. Useful for smoke tests."""
    os.makedirs(output_dir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    paths: List[str] = []
    for shard in range(num_shards):
        path = os.path.join(output_dir, f"shard-{shard:04d}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            tokens = torch.randint(0, vocab_size, (tokens_per_shard,), generator=g).tolist()
            fh.write(json.dumps({"tokens": tokens}) + "\n")
        paths.append(path)
    return paths


__all__ = [
    "PretrainingDataConfig",
    "StreamingTokenizedDataset",
    "build_pretraining_loader",
    "synthetic_token_shards",
]
