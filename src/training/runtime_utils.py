"""
Runtime helpers for model loading across GPU and CPU environments.
"""

import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

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


def _multi_gpu_max_memory() -> Optional[dict[int, str]]:
    """Reserve a little headroom on each visible GPU for activations and CUDA runtime."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None

    max_memory: dict[int, str] = {}
    reserve_bytes = 1536 * 1024 * 1024  # Leave ~1.5 GiB free per GPU.
    minimum_bytes = 2048 * 1024 * 1024

    for index in range(torch.cuda.device_count()):
        total_bytes = torch.cuda.get_device_properties(index).total_memory
        usable_bytes = max(total_bytes - reserve_bytes, minimum_bytes)
        max_memory[index] = f"{usable_bytes // (1024 * 1024)}MiB"

    return max_memory


def _normalise_quantization_mode(mode: Optional[str]) -> str:
    """Return a supported quantization mode from config or environment."""
    raw_mode = (
        os.getenv("DRL_QUANTIZATION_MODE")
        or os.getenv("DRL_QUANTIZATION")
        or mode
        or "auto"
    )
    normalised = raw_mode.strip().lower().replace("_", "-")
    aliases = {
        "": "auto",
        "true": "auto",
        "yes": "auto",
        "on": "auto",
        "false": "none",
        "no": "none",
        "off": "none",
        "disabled": "none",
        "disable": "none",
        "no-quant": "none",
        "no-quantization": "none",
        "4": "4bit",
        "4-bit": "4bit",
        "nf4": "4bit",
        "bnb-4bit": "4bit",
        "8": "8bit",
        "8-bit": "8bit",
        "int8": "8bit",
        "bnb-8bit": "8bit",
    }
    normalised = aliases.get(normalised, normalised)
    if normalised not in {"auto", "4bit", "8bit", "none"}:
        raise ValueError(
            "Unsupported quantization mode "
            f"{raw_mode!r}; expected one of auto, 4bit, 8bit, none."
        )
    if os.getenv("DRL_DISABLE_8BIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.warning(
            "DRL_DISABLE_8BIT is deprecated; treating it as DRL_QUANTIZATION_MODE=none."
        )
        return "none"
    return normalised


def _quantization_required() -> bool:
    return os.getenv("DRL_REQUIRE_QUANTIZATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configure_bnb_quantization(
    kwargs: dict[str, Any],
    *,
    mode: str,
    compute_dtype: torch.dtype,
) -> bool:
    """Attach a BitsAndBytesConfig when available; return whether k-bit loading is active."""
    if mode == "none":
        return False

    if not bitsandbytes_available():
        message = (
            "bitsandbytes quantization was requested but its CUDA backend is unavailable. "
            "Fix the Kaggle CUDA/bitsandbytes install, or set DRL_QUANTIZATION_MODE=none "
            "and ensure the GPU is fully free before training."
        )
        if _quantization_required() or mode in {"4bit", "8bit"}:
            raise RuntimeError(message)
        logger.warning("%s Falling back to standard precision model loading.", message)
        return False

    from transformers import BitsAndBytesConfig

    selected_mode = "4bit" if mode == "auto" else mode
    if selected_mode == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using bitsandbytes 4-bit NF4 loading for LoRA training.")
        return True

    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    logger.info("Using bitsandbytes 8-bit loading for LoRA training.")
    return True


def cuda_memory_snapshot() -> Optional[dict[str, float]]:
    """Return free/total CUDA memory in GiB for the active device, when available."""
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return {
        "free_gib": free_bytes / gib,
        "total_gib": total_bytes / gib,
        "used_gib": (total_bytes - free_bytes) / gib,
    }


def log_cuda_memory(label: str) -> None:
    """Log a concise CUDA memory snapshot."""
    snapshot = cuda_memory_snapshot()
    if snapshot is None:
        logger.info("%s CUDA memory: CUDA unavailable.", label)
        return
    logger.info(
        "%s CUDA memory: %.2f GiB free / %.2f GiB total (%.2f GiB used).",
        label,
        snapshot["free_gib"],
        snapshot["total_gib"],
        snapshot["used_gib"],
    )


def require_cuda_free_memory(label: str, min_free_gib: float) -> None:
    """Fail early when a previous stage is still occupying too much GPU memory."""
    snapshot = cuda_memory_snapshot()
    if snapshot is None:
        return
    if snapshot["free_gib"] < min_free_gib:
        raise RuntimeError(
            f"{label} requires at least {min_free_gib:.1f} GiB free CUDA memory, "
            f"but only {snapshot['free_gib']:.2f} GiB is free. "
            "A previous vLLM/Ray process is probably still holding the GPU; run nvidia-smi, "
            "stop those processes, or restart the Kaggle runtime before training."
        )


def build_causal_lm_load_kwargs(
    *,
    prefer_bf16: bool = False,
    allow_8bit: bool = False,
    quantization_mode: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """
    Build model loading kwargs and report whether 8-bit loading is active.
    """
    dtype = get_runtime_dtype(prefer_bf16=prefer_bf16)
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    use_kbit = False
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            kwargs["device_map"] = "balanced_low_0"
            max_memory = _multi_gpu_max_memory()
            if max_memory is not None:
                kwargs["max_memory"] = max_memory
        else:
            kwargs["device_map"] = "auto"
        if allow_8bit:
            mode = _normalise_quantization_mode(quantization_mode)
            try:
                use_kbit = _configure_bnb_quantization(
                    kwargs,
                    mode=mode,
                    compute_dtype=dtype,
                )
            except Exception as exc:
                if _quantization_required() or mode in {"4bit", "8bit"}:
                    raise
                logger.warning(
                    "Failed to configure k-bit quantization; falling back to standard precision model loading: %s",
                    exc,
                )

    return kwargs, use_kbit


def configure_training_memory(model: Any, *, gradient_checkpointing: bool) -> None:
    """Apply training-time memory settings that are easy to forget across trainers."""
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        for cfg_owner in (
            model,
            getattr(model, "model", None),
            getattr(model, "base_model", None),
        ):
            config = getattr(cfg_owner, "config", None)
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = False

        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            generation_config.use_cache = False


def is_adapter_checkpoint(model_name_or_path: str) -> bool:
    """Return True when the provided path looks like a PEFT adapter checkpoint."""
    path = Path(model_name_or_path)
    return path.exists() and (path / "adapter_config.json").exists()


def _mark_adapter_parameters_trainable(model) -> None:
    """Best-effort fallback for older PEFT versions without is_trainable support."""
    trainable_markers = ("lora_", "adapter_", "modules_to_save")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(marker in name for marker in trainable_markers)


def load_causal_lm_for_training(
    model_name_or_path: str,
    *,
    prefer_bf16: bool = False,
    allow_8bit: bool = False,
    quantization_mode: Optional[str] = None,
    device_map_override: Optional[Any] = None,
) -> tuple[Any, bool, bool, Optional[str]]:
    """
    Load either a base model or a PEFT adapter checkpoint for continued training.

    Returns:
        model: Loaded model instance.
        use_8bit: Whether 8-bit loading is active.
        loaded_adapter: True when the input path was an adapter checkpoint.
        base_model_name: Base model used for adapter checkpoints, otherwise None.
    """
    from transformers import AutoModelForCausalLM

    model_kwargs, use_8bit = build_causal_lm_load_kwargs(
        prefer_bf16=prefer_bf16,
        allow_8bit=allow_8bit,
        quantization_mode=quantization_mode,
    )
    if device_map_override is not None:
        model_kwargs.pop("device_map", None)
        model_kwargs["device_map"] = device_map_override

    if not is_adapter_checkpoint(model_name_or_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        return model, use_8bit, False, None

    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(model_name_or_path)
    base_model_name = peft_config.base_model_name_or_path
    if not base_model_name:
        raise ValueError(
            f"Adapter checkpoint at {model_name_or_path!r} is missing base_model_name_or_path."
        )

    logger.info(
        "Detected adapter checkpoint at %s; loading base model %s for continued training.",
        model_name_or_path,
        base_model_name,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs,
    )

    try:
        model = PeftModel.from_pretrained(
            base_model,
            model_name_or_path,
            is_trainable=True,
        )
    except TypeError:
        model = PeftModel.from_pretrained(base_model, model_name_or_path)
        _mark_adapter_parameters_trainable(model)

    return model, use_8bit, True, base_model_name
