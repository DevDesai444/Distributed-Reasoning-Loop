"""
Runtime helpers for model loading across GPU and CPU environments.
"""

import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _cuda_version_to_bnb_suffix(version: str | None) -> str | None:
    """Convert a CUDA version like ``12.8`` to the bitsandbytes suffix ``128``."""
    if not version:
        return None

    parts = version.strip().split(".")
    if not parts or not parts[0].isdigit():
        return None

    major = parts[0]
    minor = parts[1] if len(parts) > 1 and parts[1].isdigit() else "0"
    return f"{major}{minor}"


def _candidate_cuda_lib_dirs() -> list[Path]:
    """Collect likely CUDA library directories from the current environment."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path) -> None:
        resolved = str(path)
        if resolved in seen or not path.exists() or not path.is_dir():
            return
        seen.add(resolved)
        candidates.append(path)

    for raw_path in os.getenv("LD_LIBRARY_PATH", "").split(":"):
        if raw_path:
            add_candidate(Path(raw_path))

    for env_var in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX"):
        raw_value = os.getenv(env_var)
        if not raw_value:
            continue
        root = Path(raw_value)
        add_candidate(root / "lib64")
        add_candidate(root / "lib")

    common_roots = [
        Path("/usr/local/cuda"),
        Path("/usr/local"),
        Path("/opt/conda"),
        Path("/usr"),
    ]
    for root in common_roots:
        add_candidate(root / "lib64")
        add_candidate(root / "lib")
        if root.exists():
            for child in sorted(root.glob("cuda-*")):
                add_candidate(child / "lib64")
                add_candidate(child / "lib")

    return candidates


def _infer_cuda_suffix_from_lib_dir(lib_dir: Path) -> str | None:
    """Infer a bitsandbytes CUDA suffix from a library directory path."""
    for part in (lib_dir, *lib_dir.parents):
        name = part.name
        if not name.startswith("cuda-"):
            continue
        return _cuda_version_to_bnb_suffix(name.removeprefix("cuda-"))
    return None


def _prepend_ld_library_path(lib_dir: Path) -> None:
    """Prepend a directory to LD_LIBRARY_PATH without duplicating entries."""
    current_entries = [entry for entry in os.getenv("LD_LIBRARY_PATH", "").split(":") if entry]
    lib_dir_str = str(lib_dir)
    if current_entries and current_entries[0] == lib_dir_str:
        return

    filtered_entries = [entry for entry in current_entries if entry != lib_dir_str]
    os.environ["LD_LIBRARY_PATH"] = ":".join([lib_dir_str, *filtered_entries])


def _configure_bitsandbytes_cuda_runtime() -> None:
    """
    Best-effort configuration for bitsandbytes CUDA runtime discovery.

    bitsandbytes chooses a backend based on the CUDA version reported by PyTorch.
    On hosted notebook environments that ship CUDA runtime libraries in non-standard
    locations, explicitly wiring up LD_LIBRARY_PATH and BNB_CUDA_VERSION can make an
    otherwise valid installation usable.
    """
    if os.getenv("DRL_DISABLE_8BIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return

    desired_suffix = _cuda_version_to_bnb_suffix(torch.version.cuda)
    candidate_dirs = _candidate_cuda_lib_dirs()
    if not candidate_dirs:
        return

    matching_dir: Path | None = None
    fallback_dir: Path | None = None
    fallback_suffix: str | None = None

    for lib_dir in candidate_dirs:
        if not list(lib_dir.glob("libnvJitLink.so*")):
            continue

        inferred_suffix = _infer_cuda_suffix_from_lib_dir(lib_dir)
        if desired_suffix and inferred_suffix == desired_suffix:
            matching_dir = lib_dir
            break

        if fallback_dir is None:
            fallback_dir = lib_dir
            fallback_suffix = inferred_suffix

    chosen_dir = matching_dir or fallback_dir
    if chosen_dir is None:
        return

    _prepend_ld_library_path(chosen_dir)

    chosen_suffix = desired_suffix if matching_dir is not None else fallback_suffix
    if chosen_suffix and not os.getenv("BNB_CUDA_VERSION"):
        os.environ["BNB_CUDA_VERSION"] = chosen_suffix

    logger.info(
        "Prepared bitsandbytes CUDA runtime using %s%s",
        chosen_dir,
        f" (BNB_CUDA_VERSION={os.environ['BNB_CUDA_VERSION']})" if os.getenv("BNB_CUDA_VERSION") else "",
    )


@lru_cache(maxsize=1)
def bitsandbytes_available() -> bool:
    """Return True only when bitsandbytes is installed and its native backend loads."""
    if os.getenv("DRL_DISABLE_8BIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("8-bit loading disabled by DRL_DISABLE_8BIT.")
        return False

    _configure_bitsandbytes_cuda_runtime()

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
