"""Single-process smoke tests for FSDP utilities."""

from __future__ import annotations

import torch

from src.training.fsdp_utils import (
    FSDPConfig,
    accumulate_grad_steps,
    build_mixed_precision,
    is_distributed,
    transformer_auto_wrap_policy,
    wrap_with_fsdp,
)


def test_is_distributed_returns_false_in_single_process():
    assert is_distributed() is False


def test_wrap_with_fsdp_is_noop_without_distributed():
    model = torch.nn.Linear(4, 4)
    wrapped = wrap_with_fsdp(model)
    assert wrapped is model


def test_accumulate_grad_steps_yields_unconditionally_outside_dist():
    model = torch.nn.Linear(4, 4)
    with accumulate_grad_steps(model, is_last_microstep=False):
        x = torch.randn(2, 4)
        y = model(x)
    assert y.shape == (2, 4)


def test_mixed_precision_bf16_dtype():
    mp = build_mixed_precision("bf16")
    assert mp.param_dtype == torch.bfloat16
    assert mp.reduce_dtype == torch.float32


def test_auto_wrap_policy_callable_with_min_params_fallback():
    cfg = FSDPConfig(transformer_block_classes=())
    policy = transformer_auto_wrap_policy(cfg)
    assert callable(policy)
