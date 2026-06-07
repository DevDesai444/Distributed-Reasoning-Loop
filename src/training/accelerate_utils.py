"""
Bridge between the DRL precision/FSDP modules and Hugging Face Trainer / Accelerate.

The post-training trainers (SFT, DPO, the reward models) run on top of HF
Trainer / TRL / Accelerate. They express FSDP and mixed precision through
``TrainingArguments`` (or its TRL subclasses) rather than wrapping the model
manually. This module produces the right kwargs so SFT, DPO, and RM/PRM share
exactly the same training-systems story as the raw-loop trainers in Layer 0
and GRPO.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .precision import (
    PRECISION_BF16,
    PRECISION_FP8,
    PRECISION_FP32,
    SUPPORTED_PRECISIONS,
    env_override_precision,
    fp8_supported,
    hf_mixed_precision_kwarg,
)

logger = logging.getLogger(__name__)


_DEFAULT_FSDP_CONFIG: Dict[str, Any] = {
    "fsdp_sharding_strategy": "FULL_SHARD",
    "fsdp_backward_prefetch": "BACKWARD_PRE",
    "fsdp_forward_prefetch": False,
    "fsdp_use_orig_params": True,
    "fsdp_cpu_ram_efficient_loading": True,
    "fsdp_sync_module_states": True,
    "fsdp_activation_checkpointing": True,
    "fsdp_offload_params": False,
    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "fsdp_transformer_layer_cls_to_wrap": (
        "LlamaDecoderLayer,MistralDecoderLayer,Qwen2DecoderLayer,"
        "GPTNeoXLayer,GPT2Block,MoEDecoderLayer,TitanDecoderLayer"
    ),
}


def world_size_from_env() -> int:
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def hf_training_systems_kwargs(
    *,
    precision: Optional[str] = None,
    enable_fsdp: Optional[bool] = None,
    fsdp_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return TrainingArguments-compatible kwargs for FSDP + mixed precision.

    Args:
        precision: ``fp32`` / ``bf16`` / ``fp8``. Defaults to the value of
            ``DRL_PRECISION`` and finally ``bf16``.
        enable_fsdp: When None, FSDP is enabled if ``WORLD_SIZE`` > 1.
        fsdp_overrides: Per-field overrides merged on top of the project
            defaults.
    """
    chosen_precision = (precision or env_override_precision(PRECISION_BF16)).lower()
    if chosen_precision not in SUPPORTED_PRECISIONS:
        logger.info(
            "Unknown precision %r requested; falling back to bf16.", chosen_precision
        )
        chosen_precision = PRECISION_BF16
    if chosen_precision == PRECISION_FP8 and not fp8_supported():
        logger.info("FP8 unavailable on this runtime; using BF16 for HF Trainer.")
        chosen_precision = PRECISION_BF16

    kwargs: Dict[str, Any] = {}
    kwargs.update(hf_mixed_precision_kwarg(chosen_precision))

    fsdp_active = enable_fsdp if enable_fsdp is not None else world_size_from_env() > 1
    if fsdp_active:
        fsdp_config = dict(_DEFAULT_FSDP_CONFIG)
        if fsdp_overrides:
            fsdp_config.update(fsdp_overrides)
        kwargs["fsdp"] = "full_shard auto_wrap"
        kwargs["fsdp_config"] = fsdp_config

    kwargs["gradient_checkpointing"] = True
    kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return kwargs


def filter_supported_training_args(
    cls,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Drop keys the active TrainingArguments class does not accept.

    Different transformers/TRL versions expose different fields (notably
    ``fp8`` and the FSDP ``fsdp_*`` flags). The post-training trainers call
    this before passing kwargs into SFTConfig/DPOConfig so the upgrade works
    across the supported version matrix.
    """
    try:
        accepted = set(cls.__init__.__code__.co_varnames)
    except Exception:
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in accepted}


__all__ = [
    "hf_training_systems_kwargs",
    "filter_supported_training_args",
    "world_size_from_env",
]
