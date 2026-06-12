"""
Distributed-training helpers shared by SFT, DPO, and GRPO trainers.

Goal: pick up the torchrun environment (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`),
init a process group with the right backend, and surface a small struct the
trainers can ask "are we distributed? am I rank 0?" without each file
duplicating the same boilerplate.

NCCL is required for multi-GPU CUDA. Gloo is used for the CPU smoke run, which
is how we keep the launcher exercisable on machines with no GPU.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


@dataclass
class DistributedContext:
    """Snapshot of the current process's place in the distributed world."""

    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device
    initialized: bool

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _resolve_timeout(timeout_minutes: int) -> timedelta:
    """Mirror grpo_trainer's timeout semantics: 0 means 'no practical timeout'."""
    if timeout_minutes <= 0:
        return timedelta(days=365)
    return timedelta(minutes=timeout_minutes)


def _detect_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE")
    if raw is None:
        return 1
    try:
        size = int(raw)
    except ValueError:
        return 1
    return max(1, size)


def _detect_rank() -> int:
    raw = os.environ.get("RANK", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def _detect_local_rank() -> int:
    raw = os.environ.get("LOCAL_RANK", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def pick_backend(force_backend: Optional[str] = None) -> str:
    """Return 'nccl' on CUDA, 'gloo' otherwise. Honors an explicit override."""
    if force_backend:
        return force_backend
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_distributed(
    timeout_minutes: int = 0,
    force_backend: Optional[str] = None,
) -> DistributedContext:
    """
    Initialize the torch.distributed process group from torchrun env vars.

    Idempotent: if the group is already initialized, only the context struct is
    rebuilt. Safe to call from a single-process run (world_size=1) — it skips
    `init_process_group` and just reports a CPU/CUDA device.
    """
    world_size = _detect_world_size()
    rank = _detect_rank()
    local_rank = _detect_local_rank()
    backend = pick_backend(force_backend)

    if world_size <= 1:
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        return DistributedContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            device=device,
            initialized=False,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=_resolve_timeout(timeout_minutes),
        )
        logger.info(
            "Process group initialized rank=%s local_rank=%s world_size=%s backend=%s",
            rank,
            local_rank,
            world_size,
            backend,
        )

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        device=device,
        initialized=True,
    )


def shutdown_distributed() -> None:
    """Destroy the process group if it exists. No-op otherwise."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier_if_distributed(ctx: DistributedContext) -> None:
    """Synchronize across ranks. No-op for single-process runs."""
    if ctx.initialized and dist.is_available() and dist.is_initialized():
        dist.barrier()
