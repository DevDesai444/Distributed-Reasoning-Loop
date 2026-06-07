"""
FSDP utilities used by every distributed trainer in DRL.

Centralizes the sharded-training configuration so SFT, DPO, GRPO, the reward
models, and Layer-0 pre-training share one wrap policy, one activation-
checkpointing hook, one sharded-state-dict path, and one process-group setup.

The defaults are tuned for transformer decoders: FULL_SHARD with transformer-
block auto-wrapping, BACKWARD_PRE prefetch, activation checkpointing on
transformer blocks, and sharded optimizer state. Gradient accumulation uses
explicit ``no_sync`` boundaries via :func:`accumulate_grad_steps`.
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Type

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FSDPConfig:
    """Knobs for the FSDP wrap.

    Attributes:
        sharding_strategy: ``FULL_SHARD``, ``SHARD_GRAD_OP``, ``HYBRID_SHARD``,
            or ``NO_SHARD``.
        backward_prefetch: ``BACKWARD_PRE``, ``BACKWARD_POST``, or ``NONE``.
        cpu_offload: Offload sharded parameters to CPU between steps.
        forward_prefetch: Issue the next forward all-gather during the current
            forward — useful on slow interconnects.
        limit_all_gathers: Throttle concurrent all-gather calls.
        use_orig_params: Preserve original ``Parameter`` objects (required for
            optimizer-state-dict compatibility with HF Trainer/Accelerate).
        activation_checkpointing: Wrap transformer blocks with checkpointing.
        mixed_precision: BF16 compute / FP32 reduction by default.
        transformer_block_classes: Names or types of modules to auto-wrap.
        process_group_timeout_s: NCCL collective timeout in seconds.
    """

    sharding_strategy: str = "FULL_SHARD"
    backward_prefetch: str = "BACKWARD_PRE"
    cpu_offload: bool = False
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = True
    activation_checkpointing: bool = True
    mixed_precision: bool = True
    transformer_block_classes: Sequence[str] = field(
        default_factory=lambda: (
            "LlamaDecoderLayer",
            "MistralDecoderLayer",
            "Qwen2DecoderLayer",
            "GPTNeoXLayer",
            "GPT2Block",
            "MoEDecoderLayer",
            "TitanDecoderLayer",
        )
    )
    process_group_timeout_s: int = 1800


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def global_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return global_rank() == 0


def init_distributed(backend: str = "nccl", timeout_s: int = 1800) -> bool:
    """Initialize torch.distributed from environment variables, if not already.

    Returns True when a process group was initialized (or already running).
    Returns False when the runtime is single-process and no init was performed.
    """
    if is_distributed():
        return True
    required = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"}
    if not required.issubset(os.environ):
        return False
    chosen = backend
    if backend == "nccl" and not torch.cuda.is_available():
        chosen = "gloo"
    dist.init_process_group(
        backend=chosen,
        timeout=timedelta(seconds=timeout_s),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    logger.info(
        "Initialized torch.distributed: backend=%s world_size=%d rank=%d local_rank=%s",
        chosen,
        dist.get_world_size(),
        dist.get_rank(),
        os.environ.get("LOCAL_RANK"),
    )
    return True


def destroy_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def _resolve_block_types(modules: Iterable[str]) -> List[Type[nn.Module]]:
    resolved: List[Type[nn.Module]] = []
    seen: set[int] = set()
    for module_name in modules:
        for cls in _all_subclasses(nn.Module):
            if cls.__name__ == module_name and id(cls) not in seen:
                resolved.append(cls)
                seen.add(id(cls))
    return resolved


def _all_subclasses(cls: Type) -> Iterator[Type]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def transformer_auto_wrap_policy(config: FSDPConfig) -> Callable:
    """Return an FSDP auto-wrap policy that targets transformer blocks.

    Falls back to a min-num-params policy when no transformer block classes are
    importable in the current process (e.g. before any HF model has been
    constructed).
    """
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy as _transformer_policy,
    )

    block_types = _resolve_block_types(config.transformer_block_classes)
    if block_types:
        return functools.partial(_transformer_policy, transformer_layer_cls=set(block_types))
    return functools.partial(size_based_auto_wrap_policy, min_num_params=int(1e7))


def build_mixed_precision(precision: str = "bf16"):
    from torch.distributed.fsdp import MixedPrecision

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp8": torch.bfloat16,
    }
    compute_dtype = dtype_map.get(precision.lower(), torch.bfloat16)
    return MixedPrecision(
        param_dtype=compute_dtype,
        reduce_dtype=torch.float32,
        buffer_dtype=compute_dtype,
        cast_forward_inputs=True,
    )


def wrap_with_fsdp(
    model: nn.Module,
    *,
    config: Optional[FSDPConfig] = None,
    precision: str = "bf16",
    device_id: Optional[int] = None,
) -> nn.Module:
    """Wrap a model with FSDP using the project-wide defaults.

    Outside a distributed context the model is returned unchanged so call sites
    can be unconditional.
    """
    if not is_distributed():
        logger.info("wrap_with_fsdp called outside a distributed context; returning model unwrapped.")
        return model

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        CPUOffload,
        ShardingStrategy,
    )

    cfg = config or FSDPConfig()
    sharding_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        "NONE": None,
    }

    auto_wrap = transformer_auto_wrap_policy(cfg)
    mp = build_mixed_precision(precision) if cfg.mixed_precision else None
    sharding = sharding_map.get(cfg.sharding_strategy, ShardingStrategy.FULL_SHARD)
    prefetch = prefetch_map.get(cfg.backward_prefetch, BackwardPrefetch.BACKWARD_PRE)

    if device_id is None and torch.cuda.is_available():
        device_id = int(os.environ.get("LOCAL_RANK", "0"))

    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding,
        backward_prefetch=prefetch,
        mixed_precision=mp,
        cpu_offload=CPUOffload(offload_params=True) if cfg.cpu_offload else None,
        forward_prefetch=cfg.forward_prefetch,
        limit_all_gathers=cfg.limit_all_gathers,
        use_orig_params=cfg.use_orig_params,
        auto_wrap_policy=auto_wrap,
        device_id=device_id,
    )

    if cfg.activation_checkpointing:
        apply_activation_checkpointing(fsdp_model, cfg)

    logger.info(
        "Wrapped model with FSDP: strategy=%s prefetch=%s precision=%s ac=%s",
        cfg.sharding_strategy,
        cfg.backward_prefetch,
        precision,
        cfg.activation_checkpointing,
    )
    return fsdp_model


def apply_activation_checkpointing(model: nn.Module, config: FSDPConfig) -> None:
    """Apply activation checkpointing to the configured transformer block types."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing as _apply,
        checkpoint_wrapper,
    )

    block_types = _resolve_block_types(config.transformer_block_classes)
    if not block_types:
        logger.info("No transformer block classes resolved; activation checkpointing skipped.")
        return

    wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_fn(module: nn.Module) -> bool:
        return isinstance(module, tuple(block_types))

    _apply(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)
    logger.info(
        "Activation checkpointing applied to %s",
        ", ".join(t.__name__ for t in block_types),
    )


@contextmanager
def accumulate_grad_steps(model: nn.Module, *, is_last_microstep: bool) -> Iterator[None]:
    """Open ``no_sync`` for every micro-step except the last in an accumulation window.

    Works for both FSDP and plain modules. Outside a distributed context the
    context manager is a no-op so call sites stay simple.
    """
    if is_last_microstep or not is_distributed() or not hasattr(model, "no_sync"):
        yield
        return
    with model.no_sync():
        yield


def save_distributed_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    *,
    step: int,
    extra_state: Optional[dict] = None,
) -> None:
    """Save a sharded distributed checkpoint to ``output_dir``.

    Uses torch.distributed.checkpoint when running under FSDP so each rank
    writes its shard in parallel; falls back to a vanilla ``torch.save`` on a
    single process.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not is_distributed():
        path = os.path.join(output_dir, f"checkpoint-{step}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "extra": extra_state or {},
            },
            path,
        )
        return

    from torch.distributed.checkpoint import FileSystemWriter, save
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    state = {"model": model_state, "optimizer": optim_state, "step": step}
    if extra_state:
        state["extra"] = extra_state
    writer = FileSystemWriter(os.path.join(output_dir, f"checkpoint-{step}"))
    save(state_dict=state, storage_writer=writer)


def load_distributed_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
) -> dict:
    """Load a sharded checkpoint written by :func:`save_distributed_checkpoint`."""
    if not is_distributed():
        state = torch.load(checkpoint_dir, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        return state.get("extra", {})

    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    state = {"model": model_state, "optimizer": optim_state, "step": 0}
    reader = FileSystemReader(checkpoint_dir)
    load(state_dict=state, storage_reader=reader)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=options,
    )
    return state.get("extra", {})


@contextmanager
def maybe_no_sync(model: nn.Module, enabled: bool) -> Iterator[None]:
    """Convenience wrapper exposing ``no_sync`` only when ``enabled`` is True."""
    if enabled and hasattr(model, "no_sync") and is_distributed():
        with model.no_sync():
            yield
    else:
        with nullcontext():
            yield


__all__ = [
    "FSDPConfig",
    "init_distributed",
    "destroy_distributed",
    "is_distributed",
    "world_size",
    "global_rank",
    "is_main_process",
    "wrap_with_fsdp",
    "apply_activation_checkpointing",
    "accumulate_grad_steps",
    "save_distributed_checkpoint",
    "load_distributed_checkpoint",
    "maybe_no_sync",
    "build_mixed_precision",
    "transformer_auto_wrap_policy",
]
