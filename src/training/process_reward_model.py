"""
Process reward model for step-level reasoning supervision and reranking.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from verifier.math_verifier import MathVerifier, VerificationStatus
from verifier.step_extractor import extract_steps
from wandb_utils import ensure_wandb_run, log_to_wandb

from .reward_model import RewardModel, RewardModelConfig, load_preference_pairs
from .fsdp_utils import (
    FSDPConfig,
    init_distributed,
    is_distributed as fsdp_is_distributed,
    wrap_with_fsdp,
)
from .precision import PrecisionConfig, PrecisionPolicy, env_override_precision

logger = logging.getLogger(__name__)


@dataclass
class ProcessRewardModelConfig(RewardModelConfig):
    """Configuration for the process reward model."""

    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 2
    output_dir: str = "./outputs/process_reward_model"
    step_labels_path: str = "./synthetic_data/step_labels.jsonl"


class StepPairDataset(Dataset):
    """Pairwise step dataset for PRM training."""

    def __init__(self, data: list[dict[str, Any]], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        chosen = self.tokenizer(
            f"{item['prompt']}\n{item['chosen_step']}".strip(),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            f"{item['prompt']}\n{item['rejected_step']}".strip(),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
        }


class ProcessRewardModel(RewardModel):
    """Step-level reward model trained from diverging reasoning traces."""

    def __init__(self, config: ProcessRewardModelConfig):
        super().__init__(config)
        self.config = config

    @staticmethod
    def _verify_step(verifier: MathVerifier, step: str) -> Optional[int]:
        if "=" not in step:
            return None

        result = verifier.verify_intermediate_steps([step])[0]
        if result.status == VerificationStatus.CORRECT:
            return 1
        if result.status == VerificationStatus.INCORRECT:
            return 0
        return None

    def build_step_labels_from_preferences(
        self,
        preference_data: str | Path | Iterable[dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Build the requested step-label JSONL file from prompt/chosen/rejected pairs.
        """
        verifier = MathVerifier()
        pairs = load_preference_pairs(preference_data)
        labels: list[dict[str, Any]] = []

        for pair in pairs:
            prompt = pair["prompt"]
            chosen_steps = extract_steps(pair.get("chosen", ""))
            rejected_steps = extract_steps(pair.get("rejected", ""))

            for step_index, step in enumerate(chosen_steps):
                label = self._verify_step(verifier, step)
                if label is None:
                    continue
                labels.append(
                    {
                        "prompt": prompt,
                        "step": step,
                        "step_index": step_index,
                        "label": label,
                    }
                )

            first_error_found = False
            for step_index, step in enumerate(rejected_steps):
                label = 0 if first_error_found else self._verify_step(verifier, step)
                if label is None and not first_error_found:
                    continue
                if label == 0:
                    first_error_found = True
                labels.append(
                    {
                        "prompt": prompt,
                        "step": step,
                        "step_index": step_index,
                        "label": int(label),
                    }
                )

        target_path = Path(output_path or self.config.step_labels_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as handle:
            for record in labels:
                handle.write(json.dumps(record) + "\n")

        logger.info("Saved %s step labels to %s", len(labels), target_path)
        return labels

    def load_step_labels(
        self,
        source: str | Path | Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(source, (str, Path)):
            records: list[dict[str, Any]] = []
            with open(source) as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        return list(source)

    def build_training_pairs(
        self,
        step_labels: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Pair positive chosen steps against negative rejected steps at the same step index.
        """
        grouped: dict[tuple[str, int], dict[str, list[str]]] = defaultdict(lambda: {"positive": [], "negative": []})

        for record in step_labels:
            key = (record["prompt"], int(record["step_index"]))
            bucket = "positive" if int(record["label"]) == 1 else "negative"
            grouped[key][bucket].append(record["step"])

        pairwise_data: list[dict[str, Any]] = []
        for (prompt, step_index), values in grouped.items():
            if not values["positive"] or not values["negative"]:
                continue
            for chosen_step in values["positive"]:
                for rejected_step in values["negative"]:
                    pairwise_data.append(
                        {
                            "prompt": prompt,
                            "step_index": step_index,
                            "chosen_step": chosen_step,
                            "rejected_step": rejected_step,
                        }
                    )
        return pairwise_data

    def train_model(
        self,
        train_source: str | Path | Iterable[dict[str, Any]],
        eval_source: Optional[str | Path | Iterable[dict[str, Any]]] = None,
        *,
        source_type: str = "step_labels",
    ) -> dict[str, Any]:
        """
        Train the PRM from either step labels or raw preference pairs.
        """
        self.setup()
        ensure_wandb_run(
            project=self.config.wandb_project,
            name="process-reward-model",
            mode=self.config.wandb_mode,
            config=asdict(self.config),
            tags=["process-reward-model", "prm"],
        )

        if source_type == "preferences":
            train_labels = self.build_step_labels_from_preferences(train_source, self.config.step_labels_path)
            eval_output_path = None
            if eval_source is not None:
                eval_output_path = str(Path(self.config.step_labels_path).with_name("step_labels_eval.jsonl"))
            eval_labels = self.build_step_labels_from_preferences(eval_source, eval_output_path) if eval_source is not None else []
        else:
            train_labels = self.load_step_labels(train_source)
            eval_labels = self.load_step_labels(eval_source) if eval_source is not None else []

        train_pairs = self.build_training_pairs(train_labels)
        eval_pairs = self.build_training_pairs(eval_labels) if eval_labels else []

        if not train_pairs:
            logger.warning("No step pairs available for PRM training.")
            return {
                "epoch_losses": [],
                "num_step_labels": len(train_labels),
                "num_step_pairs": 0,
            }

        train_dataset = StepPairDataset(train_pairs, self.tokenizer, self.config.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        total_steps = max(len(train_loader) * self.config.num_epochs, 1)
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.to(self.device)
        init_distributed()
        precision_str = env_override_precision(default="bf16")
        self._precision_policy = PrecisionPolicy(PrecisionConfig(precision=precision_str))
        if fsdp_is_distributed():
            wrap_with_fsdp(
                self,
                config=FSDPConfig(activation_checkpointing=False),
                precision=precision_str,
            )

        global_step = 0
        epoch_losses: list[float] = []

        for epoch in range(self.config.num_epochs):
            self.train()
            running_loss = 0.0

            for batch in train_loader:
                global_step += 1
                batch = {key: value.to(self.device) for key, value in batch.items()}

                chosen_rewards = self.forward(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected_rewards = self.forward(batch["rejected_input_ids"], batch["rejected_attention_mask"])
                reward_margin = chosen_rewards - rejected_rewards
                loss = -F.logsigmoid(reward_margin).mean()

                optimizer.zero_grad(set_to_none=True)
                self._precision_policy.backward(loss)
                self._precision_policy.step(
                    optimizer,
                    self.parameters(),
                    step=global_step,
                )
                scheduler.step()

                running_loss += loss.item()
                log_to_wandb(
                    {
                        "prm/train_loss": float(loss.item()),
                        "prm/train_margin_mean": float(reward_margin.mean().item()),
                        "prm/learning_rate": float(scheduler.get_last_lr()[0]),
                    },
                    step=global_step,
                )

            avg_loss = running_loss / max(len(train_loader), 1)
            epoch_losses.append(avg_loss)
            log_to_wandb({"prm/epoch_loss": avg_loss, "prm/epoch": epoch + 1}, step=global_step)

            if eval_pairs:
                margins = self._evaluate_step_margins(eval_pairs)
                if margins:
                    log_to_wandb(
                        {
                            "prm/eval_margin_mean": sum(margins) / len(margins),
                        },
                        step=global_step,
                    )

        self.save(self.config.output_dir)
        return {
            "epoch_losses": epoch_losses,
            "num_step_labels": len(train_labels),
            "num_step_pairs": len(train_pairs),
        }

    def _evaluate_step_margins(self, eval_pairs: list[dict[str, Any]]) -> list[float]:
        self.eval()
        dataset = StepPairDataset(eval_pairs, self.tokenizer, self.config.max_length)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        margins: list[float] = []

        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                chosen_rewards = self.forward(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected_rewards = self.forward(batch["rejected_input_ids"], batch["rejected_attention_mask"])
                margins.extend((chosen_rewards - rejected_rewards).detach().cpu().tolist())

        return [float(margin) for margin in margins]

    def score_reasoning_steps(self, prompt: str, reasoning_text: str) -> tuple[float, list[float]]:
        """Score a reasoning trace by averaging normalized step rewards."""
        steps = extract_steps(reasoning_text)
        if not steps:
            return 0.0, []

        step_scores = [self.compute_reward(prompt, step) for step in steps]
        mean_reward = sum(step_scores) / len(step_scores)
        return mean_reward, step_scores

    def rerank_candidates(
        self,
        prompt: str,
        candidates: list[Any],
    ) -> list[tuple[Any, float]]:
        """
        Score candidates by mean step reward and return them sorted descending.
        """
        ranked: list[tuple[Any, float]] = []
        for candidate in candidates:
            reasoning = candidate if isinstance(candidate, str) else getattr(candidate, "reasoning", str(candidate))
            score, _ = self.score_reasoning_steps(prompt, reasoning)
            ranked.append((candidate, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
