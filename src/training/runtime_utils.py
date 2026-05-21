"""
Runtime helpers for model loading across GPU and CPU environments.
"""

import importlib.util
import logging
import os
from functools import lru_cache
from typing import Any

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def bitsandbytes_available() -> bool:
    """Return True only when bitsandbytes is installed and its native backend loads."""
    if os.getenv("DRL_DISABLE_8BIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("8-bit loading disabled by DRL_DISABLE_8BIT.")
        return False

    if importlib.util.find_spec("bitsandbytes") is None:
        return False

    try:
        import bitsandbytes  # noqa: F401
        from bitsandbytes import cextension as bnb_cext
    except Exception as exc:
        logger.warning(
            "bitsandbytes is installed but unavailable at runtime; disabling 8-bit loading: %s",
            exc,
        )
        return False

    if not getattr(bnb_cext, "COMPILED_WITH_CUDA", False):
        logger.warning(
            "bitsandbytes is installed but GPU quantization backend is unavailable; disabling 8-bit loading."
        )
        return False

    return True


def get_runtime_device() -> torch.device:
    """Pick the best available runtime device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_runtime_dtype(prefer_bf16: bool = False) -> torch.dtype:
    """Choose a safe dtype for the current hardware."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if prefer_bf16 else torch.float16


def build_causal_lm_load_kwargs(
    *,
    prefer_bf16: bool = False,
    allow_8bit: bool = False,
) -> tuple[dict[str, Any], bool]:
    """
    Build model loading kwargs and report whether 8-bit loading is active.
    """
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": get_runtime_dtype(prefer_bf16=prefer_bf16),
    }

    use_8bit = False
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        if allow_8bit and bitsandbytes_available():
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                use_8bit = True
            except Exception as exc:
                logger.warning(
                    "Failed to configure 8-bit quantization; falling back to standard precision model loading: %s",
                    exc,
                )
        elif allow_8bit:
            logger.warning(
                "8-bit loading unavailable; falling back to standard precision model loading."
            )

    return kwargs, use_8bit
