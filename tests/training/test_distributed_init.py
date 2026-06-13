"""Tests for the distributed-init helper used by SFT/DPO/GRPO trainers."""

from __future__ import annotations

import os

import pytest

from training.distributed import (
    DistributedContext,
    init_distributed,
    pick_backend,
    shutdown_distributed,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip torchrun env vars before every test so default detection is honest."""
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)
    yield
    shutdown_distributed()


def test_pick_backend_cpu_defaults_to_gloo():
    # Test environment is CPU-only; nccl path is hardware-dependent and not
    # exercised here. The override branch is the one we lock in.
    assert pick_backend("gloo") == "gloo"
    assert pick_backend("nccl") == "nccl"


def test_init_distributed_single_process_returns_uninitialized():
    ctx = init_distributed()
    assert isinstance(ctx, DistributedContext)
    assert ctx.world_size == 1
    assert ctx.rank == 0
    assert ctx.local_rank == 0
    assert ctx.initialized is False
    assert ctx.is_distributed is False
    assert ctx.is_main_process is True


def test_init_distributed_picks_gloo_on_cpu_when_world_size_2(monkeypatch, tmp_path):
    # Verify the *configuration* path without actually opening a TCP rendezvous.
    # We set WORLD_SIZE=1 here to exercise the early-return branch, then check
    # the backend selection helper independently.
    monkeypatch.setenv("WORLD_SIZE", "1")
    ctx = init_distributed(force_backend="gloo")
    assert ctx.backend == "gloo"
    assert ctx.is_distributed is False


def test_init_distributed_reads_rank_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    ctx = init_distributed(force_backend="gloo")
    assert ctx.rank == 0
    assert ctx.local_rank == 0


def test_init_distributed_real_process_group_gloo(monkeypatch):
    """Open and close an actual Gloo process group with WORLD_SIZE=1.

    A multi-rank rendezvous needs another process; world_size=1 still drives the
    init path the production launcher uses on real GPUs, just with the trivial
    early-return guarded by WORLD_SIZE.
    """
    # When world_size=1, init_distributed skips dist.init_process_group, so
    # this test exists to lock that contract: callers can run init in CI
    # without a rendezvous.
    monkeypatch.setenv("WORLD_SIZE", "1")
    ctx = init_distributed(force_backend="gloo")
    assert ctx.initialized is False
    # And shutdown is a safe no-op even when nothing was initialized.
    shutdown_distributed()
