"""Smoke tests for the mixed-precision subsystem."""

from __future__ import annotations

import math

import pytest
import torch

from src.training.precision import (
    PRECISION_BF16,
    PRECISION_FP32,
    PrecisionConfig,
    PrecisionPolicy,
    env_override_precision,
    hf_mixed_precision_kwarg,
)


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 4),
    )


def test_fp32_policy_runs_a_step():
    model = _tiny_model()
    policy = PrecisionPolicy(PrecisionConfig(precision=PRECISION_FP32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(2, 8)
    target = torch.randn(2, 4)
    with policy.autocast():
        loss = torch.nn.functional.mse_loss(model(x), target)
    policy.backward(loss)
    record = policy.step(optimizer, model.parameters(), step=1)
    optimizer.zero_grad(set_to_none=True)

    assert not record.overflow
    assert math.isfinite(record.pre_clip_norm)
    assert math.isfinite(record.post_clip_norm)
    assert record.post_clip_norm <= record.pre_clip_norm + 1e-6


def test_grad_clip_caps_post_norm():
    model = _tiny_model()
    config = PrecisionConfig(precision=PRECISION_FP32, grad_clip=0.01)
    policy = PrecisionPolicy(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    x = torch.randn(2, 8) * 100
    target = torch.randn(2, 4) * 100
    with policy.autocast():
        loss = torch.nn.functional.mse_loss(model(x), target)
    policy.backward(loss)
    record = policy.step(optimizer, model.parameters(), step=1)
    optimizer.zero_grad(set_to_none=True)

    assert record.post_clip_norm <= config.grad_clip + 1e-4


def test_resolved_precision_falls_back_off_cuda():
    config = PrecisionConfig(precision=PRECISION_BF16)
    assert config.resolved_precision() in {PRECISION_BF16, PRECISION_FP32}


def test_env_override_precision_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("DRL_PRECISION", "fp99")
    assert env_override_precision(default=PRECISION_BF16) == PRECISION_BF16


def test_hf_mixed_precision_kwarg_shapes():
    assert hf_mixed_precision_kwarg(PRECISION_BF16) == {"bf16": True, "fp16": False}
    assert hf_mixed_precision_kwarg(PRECISION_FP32) == {"bf16": False, "fp16": False}


def test_unsupported_precision_raises():
    with pytest.raises(ValueError):
        PrecisionConfig(precision="int4").resolved_precision()
