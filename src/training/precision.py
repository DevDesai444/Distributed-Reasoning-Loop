"""
Mixed-precision subsystem for the Distributed Reasoning Loop training stack.

A single precision policy governs every trainer (pre-training, SFT, DPO, GRPO,
reward models). The policy handles:

  - BF16 autocast compute with FP32 master weights (default path).
  - FP8 training via NVIDIA Transformer Engine on Hopper/Ada with per-tensor
    scaling and graceful fall-back to BF16 on earlier architectures.
  - Dynamic loss scaling with overflow detection, skip-and-rescale.
  - Global-norm gradient clipping with NaN/Inf guards.
  - A logged gradient-norm trace used for cross-precision stability studies.

The policy is intentionally framework-agnostic: it is consumed by raw
torch.distributed loops (Layer 0 pre-training, GRPO) and by Hugging Face
Trainer/Accelerate-driven post-training alike.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, Optional

import torch

logger = logging.getLogger(__name__)


PRECISION_FP32 = "fp32"
PRECISION_BF16 = "bf16"
PRECISION_FP8 = "fp8"
SUPPORTED_PRECISIONS = (PRECISION_FP32, PRECISION_BF16, PRECISION_FP8)


def _hopper_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _ada_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if major >= 9:
        return True
    return major == 8 and minor >= 9


def fp8_supported() -> bool:
    """True when the active device can run Transformer Engine FP8 kernels."""
    if not torch.cuda.is_available():
        return False
    if not (_hopper_or_newer() or _ada_or_newer()):
        return False
    try:
        import transformer_engine.pytorch  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class PrecisionConfig:
    """Configuration for the mixed-precision policy.

    Attributes:
        precision: One of ``fp32``, ``bf16``, ``fp8``.
        master_weights_fp32: Keep optimizer master weights in FP32.
        loss_scale_init: Initial loss scale for FP16-style scaling (used only as
            an FP8 safety net; BF16 does not need scaling).
        loss_scale_growth_factor: Multiplier when no overflow is observed for
            ``loss_scale_growth_interval`` steps.
        loss_scale_backoff_factor: Multiplier applied on overflow.
        loss_scale_growth_interval: Number of clean steps between growths.
        loss_scale_min: Lower bound on the dynamic loss scale.
        loss_scale_max: Upper bound on the dynamic loss scale.
        grad_clip: Global-norm clipping threshold.
        fp8_format: Transformer Engine FP8 format — ``hybrid`` (E4M3 forward,
            E5M2 backward) is the recommended default.
        fp8_amax_history_len: History length for FP8 amax tracking.
        fp8_amax_compute_algo: Algorithm for the FP8 amax estimator.
        log_grad_norm_every: Emit a structured grad-norm record every N steps.
    """

    precision: str = PRECISION_BF16
    master_weights_fp32: bool = True

    loss_scale_init: float = 2.0 ** 15
    loss_scale_growth_factor: float = 2.0
    loss_scale_backoff_factor: float = 0.5
    loss_scale_growth_interval: int = 2000
    loss_scale_min: float = 1.0
    loss_scale_max: float = 2.0 ** 24

    grad_clip: float = 1.0

    fp8_format: str = "hybrid"
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"

    log_grad_norm_every: int = 50

    def resolved_precision(self) -> str:
        precision = self.precision.lower().strip()
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision {self.precision!r}; expected one of {SUPPORTED_PRECISIONS}."
            )
        if precision == PRECISION_FP8 and not fp8_supported():
            logger.info(
                "FP8 requested but the runtime does not expose Transformer Engine FP8; "
                "falling back to BF16."
            )
            return PRECISION_BF16
        if precision == PRECISION_BF16 and not torch.cuda.is_available():
            return PRECISION_FP32
        return precision


@dataclass
class GradientNormRecord:
    step: int
    pre_clip_norm: float
    post_clip_norm: float
    loss_scale: float
    overflow: bool


class _DynamicLossScaler:
    """A minimal dynamic loss scaler.

    BF16 does not require loss scaling, but we still use this object to keep a
    consistent step interface across precisions and to act as a safety net for
    FP8 training where backward-pass amax tracking can occasionally produce
    near-overflow values.
    """

    def __init__(self, config: PrecisionConfig):
        self._scale = float(config.loss_scale_init)
        self._growth = config.loss_scale_growth_factor
        self._backoff = config.loss_scale_backoff_factor
        self._interval = max(1, config.loss_scale_growth_interval)
        self._min = config.loss_scale_min
        self._max = config.loss_scale_max
        self._clean_steps = 0

    @property
    def scale(self) -> float:
        return self._scale

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self._scale == 1.0:
            return loss
        return loss * self._scale

    def unscale_(self, parameters: Iterable[torch.nn.Parameter]) -> bool:
        """In-place unscale; returns True if any non-finite gradient is found."""
        inv = 1.0 / self._scale if self._scale != 0.0 else 1.0
        found_inf = False
        for param in parameters:
            if param.grad is None:
                continue
            if inv != 1.0:
                param.grad.detach().mul_(inv)
            if not torch.isfinite(param.grad).all():
                found_inf = True
        return found_inf

    def update(self, overflow: bool) -> None:
        if overflow:
            self._scale = max(self._scale * self._backoff, self._min)
            self._clean_steps = 0
            return
        self._clean_steps += 1
        if self._clean_steps >= self._interval:
            self._scale = min(self._scale * self._growth, self._max)
            self._clean_steps = 0


class PrecisionPolicy:
    """The runtime entry point for mixed precision in DRL trainers.

    Typical use::

        policy = PrecisionPolicy(PrecisionConfig(precision="bf16"))
        for step, batch in enumerate(loader):
            with policy.autocast():
                loss = model(**batch).loss
            policy.backward(loss)
            policy.step(optimizer, model.parameters(), step=step)
            optimizer.zero_grad(set_to_none=True)
    """

    def __init__(self, config: Optional[PrecisionConfig] = None):
        self.config = config or PrecisionConfig()
        self._precision = self.config.resolved_precision()
        self._scaler = _DynamicLossScaler(self.config)
        self._grad_norm_history: List[GradientNormRecord] = []
        self._fp8_recipe = None
        if self._precision == PRECISION_FP8:
            self._fp8_recipe = self._build_fp8_recipe()

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def autocast_dtype(self) -> torch.dtype:
        if self._precision in (PRECISION_BF16, PRECISION_FP8):
            return torch.bfloat16
        return torch.float32

    @property
    def grad_norm_history(self) -> List[GradientNormRecord]:
        return list(self._grad_norm_history)

    @contextmanager
    def autocast(self) -> Iterator[None]:
        """Open the appropriate autocast region for the active precision."""
        if self._precision == PRECISION_FP32 or not torch.cuda.is_available():
            yield
            return

        amp_ctx = torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        if self._precision == PRECISION_FP8 and self._fp8_recipe is not None:
            try:
                import transformer_engine.pytorch as te
            except Exception:
                with amp_ctx:
                    yield
                return
            with amp_ctx, te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe):
                yield
            return
        with amp_ctx:
            yield

    def backward(self, loss: torch.Tensor) -> None:
        if not torch.is_tensor(loss):
            raise TypeError("PrecisionPolicy.backward expected a tensor loss.")
        scaled = self._scaler.scale_loss(loss)
        scaled.backward()

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.nn.Parameter],
        *,
        step: int = 0,
        sync_grad: bool = True,
    ) -> GradientNormRecord:
        """Unscale, clip, optimizer.step() with overflow handling.

        Returns a structured record so callers can log or aggregate per-step
        gradient norms across the run.
        """
        params = [p for p in parameters if p.requires_grad]
        if not sync_grad:
            return GradientNormRecord(
                step=step,
                pre_clip_norm=float("nan"),
                post_clip_norm=float("nan"),
                loss_scale=self._scaler.scale,
                overflow=False,
            )

        overflow = self._scaler.unscale_(params)
        if overflow:
            self._scaler.update(overflow=True)
            record = GradientNormRecord(
                step=step,
                pre_clip_norm=float("inf"),
                post_clip_norm=float("inf"),
                loss_scale=self._scaler.scale,
                overflow=True,
            )
            self._record(record)
            return record

        pre_clip = self._global_norm(params)
        if not math.isfinite(pre_clip):
            self._scaler.update(overflow=True)
            record = GradientNormRecord(
                step=step,
                pre_clip_norm=pre_clip,
                post_clip_norm=pre_clip,
                loss_scale=self._scaler.scale,
                overflow=True,
            )
            self._record(record)
            return record

        torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
        post_clip = self._global_norm(params)
        optimizer.step()
        self._scaler.update(overflow=False)
        record = GradientNormRecord(
            step=step,
            pre_clip_norm=pre_clip,
            post_clip_norm=post_clip,
            loss_scale=self._scaler.scale,
            overflow=False,
        )
        self._record(record)
        return record

    def state_dict(self) -> dict:
        return {
            "precision": self._precision,
            "loss_scale": self._scaler.scale,
            "config": self.config.__dict__,
        }

    def _record(self, record: GradientNormRecord) -> None:
        self._grad_norm_history.append(record)
        if (
            self.config.log_grad_norm_every > 0
            and record.step > 0
            and record.step % self.config.log_grad_norm_every == 0
        ):
            logger.info(
                "grad_norm step=%d pre=%.4f post=%.4f loss_scale=%.1f overflow=%s",
                record.step,
                record.pre_clip_norm,
                record.post_clip_norm,
                record.loss_scale,
                record.overflow,
            )

    def _build_fp8_recipe(self) -> Optional[Any]:
        try:
            from transformer_engine.common.recipe import DelayedScaling, Format
        except Exception:
            logger.info("Transformer Engine FP8 recipe import failed; FP8 path inactive.")
            return None
        fmt = Format.HYBRID if self.config.fp8_format.lower() == "hybrid" else Format.E4M3
        return DelayedScaling(
            fp8_format=fmt,
            amax_history_len=self.config.fp8_amax_history_len,
            amax_compute_algo=self.config.fp8_amax_compute_algo,
        )

    @staticmethod
    def _global_norm(parameters: List[torch.nn.Parameter]) -> float:
        total_sq = torch.zeros(1, device="cpu")
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total_sq += grad.pow(2).sum().to("cpu", non_blocking=False)
        return float(total_sq.sqrt().item())


def hf_mixed_precision_kwarg(precision: str) -> dict:
    """Translate the DRL precision string to Hugging Face TrainingArguments flags."""
    resolved = precision.lower()
    if resolved == PRECISION_BF16:
        return {"bf16": True, "fp16": False}
    if resolved == PRECISION_FP8:
        kwargs = {"bf16": True, "fp16": False}
        if hasattr(_HFArgsCompat, "fp8"):
            kwargs["fp8"] = True
        return kwargs
    return {"bf16": False, "fp16": False}


class _HFArgsCompat:
    """Probe to discover which TrainingArguments fields exist on this transformers version."""

    fp8 = False
    try:
        from transformers import TrainingArguments  # type: ignore

        fp8 = "fp8" in TrainingArguments.__init__.__code__.co_varnames
    except Exception:
        fp8 = False


def env_override_precision(default: str = PRECISION_BF16) -> str:
    """Read DRL_PRECISION from the environment; fall back to the supplied default."""
    raw = os.getenv("DRL_PRECISION", "").strip().lower()
    if raw in SUPPORTED_PRECISIONS:
        return raw
    return default


__all__ = [
    "PrecisionConfig",
    "PrecisionPolicy",
    "GradientNormRecord",
    "PRECISION_FP32",
    "PRECISION_BF16",
    "PRECISION_FP8",
    "SUPPORTED_PRECISIONS",
    "fp8_supported",
    "hf_mixed_precision_kwarg",
    "env_override_precision",
]
