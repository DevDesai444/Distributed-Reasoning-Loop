"""
Small helpers for optional Weights & Biases integration.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def get_wandb():
    """Return the wandb module when available."""
    return wandb


def ensure_wandb_run(
    *,
    project: str,
    name: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    mode: Optional[str] = None,
    tags: Optional[list[str]] = None,
):
    """
    Create or update a W&B run.

    Defaults to offline mode so local development does not depend on network access.
    """
    if wandb is None:
        return None

    if wandb.run is None:
        init_kwargs = {
            "project": project,
            "name": name,
            "mode": mode or os.getenv("WANDB_MODE", "offline"),
            "reinit": True,
        }
        if tags:
            init_kwargs["tags"] = tags

        try:
            wandb.init(**init_kwargs)
        except Exception as exc:  # pragma: no cover - network/login dependent
            logger.warning("Failed to initialize W&B: %s", exc)
            return None

    if config:
        try:
            wandb.config.update(config, allow_val_change=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to update W&B config: %s", exc)

    return wandb.run


def log_to_wandb(data: dict[str, Any], step: Optional[int] = None):
    """Log a metric payload when W&B is active."""
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.log(data, step=step)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to log to W&B: %s", exc)


def make_histogram(values: Iterable[float]):
    """Create a W&B histogram when available."""
    if wandb is None:
        return list(values)
    try:
        return wandb.Histogram(list(values))
    except Exception:  # pragma: no cover - defensive
        return list(values)


def finish_wandb():
    """Close the active W&B run if one exists."""
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to finish W&B run: %s", exc)
