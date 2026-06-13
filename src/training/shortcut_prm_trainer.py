"""
Trainer for the shortcut process reward model.

Trains a sequence-classification head over an HF encoder on the JSONL splits
produced by :mod:`src.training.shortcut_prm_dataset`. Reports accuracy, F1,
and AUC after each epoch to stdout and to ``train_log.jsonl`` inside the
output directory.

The trainer has a ``cpu_only`` smoke path that swaps the backbone for a
random-initialized tiny BERT-style encoder. This makes the loop runnable on
machines with no GPU and no HuggingFace Hub access.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .shortcut_prm import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_LENGTH,
    ShortcutPRM,
    ShortcutPRMConfig,
)

logger = logging.getLogger(__name__)


SMOKE_MAX_LENGTH = 128


@dataclass
class TrainConfig:
    """Configuration for the shortcut-PRM training loop."""

    dataset_dir: str = ""
    output_dir: str = "./outputs/shortcut_prm_model"
    model_name: str = DEFAULT_BASE_MODEL
    n_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    max_length: int = DEFAULT_MAX_LENGTH
    seed: int = 17
    cpu_only: bool = False
    smoke_max_steps: int = 3
    weight_decay: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    eval_loss: float
    eval_accuracy: float
    eval_f1: float
    eval_auc: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    output_dir: str
    train_log_path: str
    final_metrics: Optional[EpochMetrics]
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "train_log_path": self.train_log_path,
            "final_metrics": self.final_metrics.to_dict() if self.final_metrics else None,
            "epoch_metrics": [m.to_dict() for m in self.epoch_metrics],
            "config": self.config,
        }


# ----------------------------------------------------------------- dataset


class ShortcutPRMDataset(Dataset):
    """Reads a JSONL file emitted by :mod:`shortcut_prm_dataset`."""

    def __init__(self, path: Path, tokenizer: Any, max_length: int):
        self.records = self._load(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _load(path: Path) -> List[dict[str, Any]]:
        rows: List[dict[str, Any]] = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.records[idx]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


# ----------------------------------------------------------------- metrics


def _accuracy(preds: List[int], labels: List[int]) -> float:
    if not preds:
        return 0.0
    correct = sum(int(p == l) for p, l in zip(preds, labels))
    return correct / len(preds)


def _f1(preds: List[int], labels: List[int]) -> float:
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _auc(scores: List[float], labels: List[int]) -> float:
    """ROC AUC computed via the Mann-Whitney U identity. No sklearn dependency."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


# ----------------------------------------------------------------- backbone


def _build_smoke_model(num_labels: int = 2) -> tuple[Any, Any, ShortcutPRMConfig]:
    """
    Build a tiny random-init BERT-style encoder for the CPU smoke path.

    Avoids any HuggingFace Hub download. We also build a trivial whitespace
    tokenizer if ``bert-base-uncased`` isn't already cached locally.
    """
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=30522,
        max_position_embeddings=SMOKE_MAX_LENGTH,
        type_vocab_size=2,
        num_labels=num_labels,
    )
    model = BertForSequenceClassification(config)
    tokenizer = _build_smoke_tokenizer(model_max_length=SMOKE_MAX_LENGTH)
    prm_config = ShortcutPRMConfig(
        model_name="smoke-bert",
        max_length=SMOKE_MAX_LENGTH,
        num_labels=num_labels,
    )
    return model, tokenizer, prm_config


def _build_smoke_tokenizer(model_max_length: int) -> Any:
    """Prefer a cached bert tokenizer; fall back to a trivial whitespace one."""
    try:
        from transformers import BertTokenizerFast

        tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        tokenizer.model_max_length = model_max_length
        return tokenizer
    except Exception as exc:  # pragma: no cover - fallback for fully offline boxes
        logger.warning("Cached BERT tokenizer not available (%s); using trivial fallback", exc)
        return _TrivialWhitespaceTokenizer(model_max_length=model_max_length)


class _TrivialWhitespaceTokenizer:
    """
    Minimal tokenizer that mimics the small surface of an HF tokenizer used
    by the trainer. Splits on whitespace, hashes tokens into the vocab range
    of the smoke encoder, and supports the keyword arguments the trainer
    actually passes.
    """

    def __init__(self, model_max_length: int, vocab_size: int = 30522):
        self.model_max_length = model_max_length
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102

    def _encode(self, text: str) -> List[int]:
        ids = [self.cls_token_id]
        for token in (text or "").split():
            ids.append((hash(token) % (self.vocab_size - 200)) + 200)
        ids.append(self.sep_token_id)
        return ids

    def __call__(
        self,
        text: str | List[str],
        *,
        truncation: bool = True,
        max_length: Optional[int] = None,
        padding: Any = True,
        return_tensors: Optional[str] = None,
    ) -> dict[str, Any]:
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = list(text)
            single = False

        max_len = max_length or self.model_max_length
        all_ids = [self._encode(t) for t in texts]
        if truncation:
            all_ids = [ids[:max_len] for ids in all_ids]

        if padding == "max_length":
            pad_to = max_len
        elif padding:
            pad_to = max(len(ids) for ids in all_ids) if all_ids else 0
        else:
            pad_to = max(len(ids) for ids in all_ids) if all_ids else 0

        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for ids in all_ids:
            length = len(ids)
            pad_len = max(0, pad_to - length)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * length + [0] * pad_len)

        if return_tensors == "pt":
            t_input = torch.tensor(input_ids, dtype=torch.long)
            t_mask = torch.tensor(attention_mask, dtype=torch.long)
            if single:
                t_input = t_input.unsqueeze(0) if t_input.ndim == 1 else t_input
                t_mask = t_mask.unsqueeze(0) if t_mask.ndim == 1 else t_mask
            return {"input_ids": t_input, "attention_mask": t_mask}

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_max_length": self.model_max_length,
            "vocab_size": self.vocab_size,
            "kind": "trivial_whitespace",
        }
        (out / "trivial_tokenizer.json").write_text(json.dumps(payload, indent=2))


# ------------------------------------------------------------------- train


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _eval_model(
    model: ShortcutPRM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    losses: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_scores: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            losses.append(float(outputs.loss.item()))
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(int(p) for p in preds.detach().cpu().tolist())
            all_labels.extend(int(l) for l in batch["labels"].detach().cpu().tolist())
            all_scores.extend(float(s) for s in probs.detach().cpu().tolist())
    mean_loss = sum(losses) / max(len(losses), 1)
    return (
        mean_loss,
        _accuracy(all_preds, all_labels),
        _f1(all_preds, all_labels),
        _auc(all_scores, all_labels),
    )


def train_shortcut_prm(config: TrainConfig) -> TrainResult:
    """
    Train a shortcut PRM. Returns the final eval metrics and the saved path.

    When ``config.cpu_only`` is True, the trainer uses a random-init tiny
    encoder so the full loop runs without any network access. It also caps
    training at ``config.smoke_max_steps`` steps so the smoke completes in
    seconds.
    """
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = out_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()

    _set_seed(config.seed)
    device = torch.device("cpu") if config.cpu_only else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if config.cpu_only:
        backbone, tokenizer, prm_config = _build_smoke_model()
        prm = ShortcutPRM(config=prm_config, tokenizer=tokenizer, backbone=backbone)
        effective_max_length = prm_config.max_length
    else:
        prm = ShortcutPRM.from_pretrained(config.model_name)
        prm.config.max_length = config.max_length
        effective_max_length = config.max_length

    prm.to(device)

    dataset_dir = Path(config.dataset_dir)
    train_path = dataset_dir / "train.jsonl"
    eval_path = dataset_dir / "eval.jsonl"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Expected train.jsonl and eval.jsonl inside {dataset_dir}"
        )

    train_dataset = ShortcutPRMDataset(train_path, prm.tokenizer, effective_max_length)
    eval_dataset = ShortcutPRMDataset(eval_path, prm.tokenizer, effective_max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size)

    optimizer = torch.optim.AdamW(
        prm.backbone.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epoch_metrics: List[EpochMetrics] = []
    global_step = 0
    cancelled = False

    for epoch in range(1, config.n_epochs + 1):
        prm.train()
        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            if config.cpu_only and global_step >= config.smoke_max_steps:
                cancelled = True
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = prm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1
            global_step += 1

        train_loss = running_loss / max(steps, 1)
        eval_loss, accuracy, f1, auc = _eval_model(prm, eval_loader, device)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            eval_loss=eval_loss,
            eval_accuracy=accuracy,
            eval_f1=f1,
            eval_auc=auc,
        )
        epoch_metrics.append(metrics)
        with open(train_log_path, "a") as handle:
            handle.write(json.dumps(metrics.to_dict()) + "\n")
        logger.info(
            "epoch=%d train_loss=%.4f eval_loss=%.4f acc=%.3f f1=%.3f auc=%.3f",
            epoch,
            train_loss,
            eval_loss,
            accuracy,
            f1,
            auc,
        )
        if cancelled:
            break

    prm.save_pretrained(out_dir)
    with open(out_dir / "train_config.json", "w") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    return TrainResult(
        output_dir=str(out_dir),
        train_log_path=str(train_log_path),
        final_metrics=epoch_metrics[-1] if epoch_metrics else None,
        epoch_metrics=epoch_metrics,
        config=config.to_dict(),
    )
