"""
Shortcut process reward model.

A small binary-classification encoder that takes a reasoning trace as input
and returns the probability that the trace is a "shortcut" — verifier
accepts, judge rejects. Trained from the shortcut bucket emitted by the
agreement framework.

Designed to be plugged into the agreement framework as a 4th evaluator
alongside the verifier, the LLM judge, and the ORM.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


DEFAULT_BASE_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 512


@dataclass
class ShortcutPRMConfig:
    """Light wrapper config for the shortcut PRM."""

    model_name: str = DEFAULT_BASE_MODEL
    max_length: int = DEFAULT_MAX_LENGTH
    num_labels: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShortcutPRM(nn.Module):
    """
    HuggingFace sequence-classification head over a transformer encoder.

    Exposes:
      - `from_pretrained(name_or_path)` to load a base encoder or a saved
        checkpoint (mirrors the HF interface).
      - `save_pretrained(path)` to persist tokenizer + classifier together.
      - `score(text)` / `score_batch(texts)` for inference; both return the
        probability of label 1 (the "shortcut" class).
    """

    def __init__(
        self,
        config: Optional[ShortcutPRMConfig] = None,
        *,
        tokenizer: Any = None,
        backbone: Any = None,
    ):
        super().__init__()
        self.config = config or ShortcutPRMConfig()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self._device = torch.device("cpu")

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        config: Optional[ShortcutPRMConfig] = None,
        tokenizer: Any = None,
        backbone: Any = None,
    ) -> "ShortcutPRM":
        """Load a base encoder (or saved PRM) and wrap it with a classification head."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        resolved_config = config or ShortcutPRMConfig(model_name=name_or_path)

        tok = tokenizer
        if tok is None:
            tok = AutoTokenizer.from_pretrained(name_or_path)

        model = backbone
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                name_or_path,
                num_labels=resolved_config.num_labels,
            )

        instance = cls(config=resolved_config, tokenizer=tok, backbone=model)
        return instance

    def save_pretrained(self, path: str | Path) -> Path:
        """Persist tokenizer, backbone, and a small config blob to ``path``."""
        if self.backbone is None or self.tokenizer is None:
            raise RuntimeError("ShortcutPRM must be initialized before save_pretrained")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        with open(out_dir / "shortcut_prm_config.json", "w") as handle:
            json.dump(self.config.to_dict(), handle, indent=2)
        return out_dir

    @classmethod
    def load(cls, path: str | Path) -> "ShortcutPRM":
        """Load a previously saved ShortcutPRM directory."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        load_path = Path(path)
        cfg_path = load_path / "shortcut_prm_config.json"
        if cfg_path.exists():
            with open(cfg_path) as handle:
                cfg_dict = json.load(handle)
            config = ShortcutPRMConfig(**cfg_dict)
        else:
            config = ShortcutPRMConfig(model_name=str(load_path))
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        backbone = AutoModelForSequenceClassification.from_pretrained(load_path)
        return cls(config=config, tokenizer=tokenizer, backbone=backbone)

    # ------------------------------------------------------------- forward

    def to(self, *args, **kwargs):  # type: ignore[override]
        result = super().to(*args, **kwargs)
        if self.backbone is not None:
            self.backbone = self.backbone.to(*args, **kwargs)
        # Track device so inference helpers can move tensors automatically.
        for arg in args:
            if isinstance(arg, torch.device):
                self._device = arg
            elif isinstance(arg, str):
                self._device = torch.device(arg)
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        if self.backbone is None:
            raise RuntimeError("ShortcutPRM backbone is not initialized")
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ----------------------------------------------------------- inference

    def _tokenize(self, texts: List[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("ShortcutPRM tokenizer is not initialized")
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.backbone.parameters()).device if self.backbone is not None else self._device
        return {key: value.to(device) for key, value in encodings.items()}

    @torch.no_grad()
    def score_batch(self, traces: List[str]) -> List[float]:
        """Return shortcut probability in [0, 1] for each trace."""
        if not traces:
            return []
        if self.backbone is None:
            raise RuntimeError("ShortcutPRM backbone is not initialized")
        self.backbone.eval()
        inputs = self._tokenize(list(traces))
        outputs = self.backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
        else:
            probs = torch.softmax(logits, dim=-1)[:, 1]
        return [float(p) for p in probs.detach().cpu().tolist()]

    def score(self, trace_text: str) -> float:
        """Single-sample convenience around :meth:`score_batch`."""
        return self.score_batch([trace_text])[0]
