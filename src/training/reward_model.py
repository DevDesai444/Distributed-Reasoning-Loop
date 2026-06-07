"""
Learned outcome reward model trained with Bradley-Terry pairwise loss.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .runtime_utils import get_runtime_device, get_runtime_dtype
from .fsdp_utils import (
    FSDPConfig,
    init_distributed,
    is_distributed as fsdp_is_distributed,
    wrap_with_fsdp,
)
from .precision import PrecisionConfig, PrecisionPolicy, env_override_precision
from wandb_utils import ensure_wandb_run, log_to_wandb, make_histogram

logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration for the learned outcome reward model."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = 1024
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "./outputs/reward_model"
    wandb_project: str = "distributed-reasoning-loop"
    wandb_mode: str = "offline"


def load_preference_pairs(source: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load preference pairs from JSONL or pass through an in-memory iterable."""
    if isinstance(source, (str, Path)):
        pairs: list[dict[str, Any]] = []
        with open(source) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs
    return list(source)


class PreferencePairDataset(Dataset):
    """Dataset that tokenizes chosen/rejected prompt-response pairs."""

    def __init__(self, data: list[dict[str, Any]], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def _format_example(self, prompt: str, response: str) -> str:
        return f"{prompt}\n{response}".strip()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        chosen_tokens = self.tokenizer(
            self._format_example(item["prompt"], item["chosen"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_tokens = self.tokenizer(
            self._format_example(item["prompt"], item["rejected"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
        }


class RewardModel(nn.Module):
    """
    Outcome reward model that scores a full prompt/response pair with a scalar.

    The backbone is a pretrained transformer encoder-style representation built
    from the Qwen causal model hidden states, with a linear scalar head on top
    of the first token representation as the requested CLS-equivalent.
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        self.tokenizer = None
        self.backbone = None
        self.reward_head: Optional[nn.Linear] = None
        self.device = get_runtime_device()

    def setup(self):
        """Load the tokenizer, backbone, and scalar reward head."""
        if self.backbone is not None and self.reward_head is not None and self.tokenizer is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=get_runtime_dtype(prefer_bf16=True),
        )
        if not torch.cuda.is_available():
            self.backbone.to(self.device)

        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        self.reward_head.to(next(self.backbone.parameters()).device)
        logger.info("Reward model initialized with %s", self.config.model_name)

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        return hidden_states[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.setup()
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self._pool_hidden_states(outputs.last_hidden_state, attention_mask)
        return self.reward_head(pooled).squeeze(-1)

    def _tokenize_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        self.setup()
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        return {key: value.to(device) for key, value in encodings.items()}

    def compute_raw_reward(self, prompt: str, response: str) -> float:
        """Return the unnormalized scalar reward."""
        self.eval()
        inputs = self._tokenize_texts([f"{prompt}\n{response}".strip()])
        with torch.no_grad():
            reward = self.forward(inputs["input_ids"], inputs["attention_mask"])
        return float(reward.item())

    def compute_reward(self, prompt: str, response: str) -> float:
        """Return a normalized reward in [0, 1]."""
        raw_reward = self.compute_raw_reward(prompt, response)
        return float(torch.sigmoid(torch.tensor(raw_reward)).item())

    def score_batch(
        self,
        prompts: list[str],
        responses: list[str],
        normalize: bool = True,
    ) -> list[float]:
        """Score a batch of prompt/response pairs."""
        self.eval()
        texts = [f"{prompt}\n{response}".strip() for prompt, response in zip(prompts, responses)]
        inputs = self._tokenize_texts(texts)
        with torch.no_grad():
            rewards = self.forward(inputs["input_ids"], inputs["attention_mask"])
        if normalize:
            rewards = torch.sigmoid(rewards)
        return [float(value) for value in rewards.detach().cpu().tolist()]

    def _evaluate_margins(self, eval_data: list[dict[str, Any]]) -> list[float]:
        """Compute reward margins on an evaluation split."""
        if not eval_data:
            return []

        self.eval()
        margins: list[float] = []
        eval_dataset = PreferencePairDataset(eval_data, self.tokenizer, self.config.max_length)
        eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False)
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in eval_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                chosen_rewards = self.forward(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected_rewards = self.forward(batch["rejected_input_ids"], batch["rejected_attention_mask"])
                margins.extend((chosen_rewards - rejected_rewards).detach().cpu().tolist())

        return [float(margin) for margin in margins]

    def train_model(
        self,
        train_data: str | Path | Iterable[dict[str, Any]],
        eval_data: Optional[str | Path | Iterable[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Train the reward model on prompt/chosen/rejected preference pairs.
        """
        self.setup()

        train_pairs = load_preference_pairs(train_data)
        eval_pairs = load_preference_pairs(eval_data) if eval_data is not None else []

        if not train_pairs:
            logger.warning("No preference pairs provided for reward-model training.")
            return {"epoch_losses": [], "num_train_pairs": 0, "num_eval_pairs": len(eval_pairs)}

        train_dataset = PreferencePairDataset(train_pairs, self.tokenizer, self.config.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

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
        ensure_wandb_run(
            project=self.config.wandb_project,
            name="reward-model",
            mode=self.config.wandb_mode,
            config=asdict(self.config),
            tags=["reward-model", "bradley-terry"],
        )

        global_step = 0
        epoch_losses: list[float] = []

        for epoch in range(self.config.num_epochs):
            self.train()
            running_loss = 0.0

            for batch in train_loader:
                global_step += 1
                batch = {key: value.to(self.device) for key, value in batch.items()}

                chosen_rewards = self.forward(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                )
                rejected_rewards = self.forward(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                )

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
                        "reward_model/train_loss": float(loss.item()),
                        "reward_model/train_margin_mean": float(reward_margin.mean().item()),
                        "reward_model/learning_rate": float(scheduler.get_last_lr()[0]),
                    },
                    step=global_step,
                )

            avg_loss = running_loss / max(len(train_loader), 1)
            epoch_losses.append(avg_loss)
            logger.info(
                "Reward model epoch %s/%s complete. Loss: %.4f",
                epoch + 1,
                self.config.num_epochs,
                avg_loss,
            )

            margins = self._evaluate_margins(eval_pairs) if eval_pairs else []
            log_payload = {
                "reward_model/epoch": epoch + 1,
                "reward_model/epoch_loss": avg_loss,
            }
            if margins:
                log_payload["reward_model/eval_margin_histogram"] = make_histogram(margins)
                log_payload["reward_model/eval_margin_mean"] = sum(margins) / len(margins)
            log_to_wandb(log_payload, step=global_step)

        self.save(self.config.output_dir)
        return {
            "epoch_losses": epoch_losses,
            "num_train_pairs": len(train_pairs),
            "num_eval_pairs": len(eval_pairs),
        }

    def save(self, path: str):
        """Save the backbone, tokenizer, reward head, and config."""
        self.setup()
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.backbone.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        torch.save(self.reward_head.state_dict(), output_path / "reward_head.pt")

        with open(output_path / "reward_model_config.json", "w") as handle:
            json.dump(asdict(self.config), handle, indent=2)

        logger.info("Reward model saved to %s", output_path)

    def load(self, path: str):
        """Load a saved reward model checkpoint."""
        checkpoint_path = Path(path)
        config_path = checkpoint_path / "reward_model_config.json"
        if config_path.exists():
            with open(config_path) as handle:
                saved_config = json.load(handle)
            self.config = type(self.config)(**saved_config)

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModel.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=get_runtime_dtype(prefer_bf16=True),
        )
        if not torch.cuda.is_available():
            self.backbone.to(self.device)

        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        state_dict = torch.load(checkpoint_path / "reward_head.pt", map_location=self.device)
        self.reward_head.load_state_dict(state_dict)
        self.reward_head.to(next(self.backbone.parameters()).device)
        self.eval()
        logger.info("Reward model loaded from %s", checkpoint_path)
