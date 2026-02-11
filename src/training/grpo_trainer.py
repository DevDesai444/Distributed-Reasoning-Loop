"""
Group Relative Policy Optimization (GRPO) trainer with verifier-aware logging.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from verifier import create_verifier, get_default_sandbox_image
from wandb_utils import ensure_wandb_run, log_to_wandb

from .runtime_utils import build_causal_lm_load_kwargs, get_runtime_device

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    group_size: int = 2
    kl_coef: float = 0.1
    clip_range: float = 0.2
    kl_threshold: float = 0.1

    learning_rate: float = 5e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    max_length: int = 1024
    max_prompt_length: int = 256

    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    logging_steps: int = 10
    eval_interval_steps: int = 50
    heldout_eval_size: int = 20
    heldout_split: str = "test"
    eval_max_new_tokens: int = 256
    output_dir: str = "./grpo_output"

    bf16: bool = True
    gradient_checkpointing: bool = True

    verifier_type: str = "math"
    verifier_timeout: int = 10
    code_docker_image: str = field(default_factory=get_default_sandbox_image)
    code_memory_limit: str = "512m"

    wandb_project: str = "distributed-reasoning-loop"
    wandb_mode: str = "offline"


class GRPODataset(Dataset):
    """Dataset for GRPO training with grouped prompt responses."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.groups = self._create_groups(data)

    def _create_groups(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt_to_responses: dict[str, dict[str, Any]] = {}

        for item in data:
            if "chosen" in item and "rejected" in item:
                prompt = item.get("prompt", "")
                bucket = prompt_to_responses.setdefault(
                    prompt,
                    {"prompt": prompt, "chosen": [], "rejected": []},
                )
                bucket["chosen"].append(item["chosen"])
                bucket["rejected"].append(item["rejected"])
                continue

            prompt = item.get("prompt", item.get("problem", ""))
            bucket = prompt_to_responses.setdefault(
                prompt,
                {"prompt": prompt, "chosen": [], "rejected": []},
            )

            response = item.get("reasoning", item.get("response", ""))
            if item.get("is_correct", False):
                bucket["chosen"].append(response)
            else:
                bucket["rejected"].append(response)

        groups = [group for group in prompt_to_responses.values() if group["chosen"] and group["rejected"]]
        logger.info("Created %s GRPO prompt groups", len(groups))
        return groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.groups[idx]


class ReasoningGRPOTrainer:
    """Custom GRPO trainer with offline verifier-backed evaluation."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.device = get_runtime_device()
        self.use_kbit_training = False
        self.verifier = None
        self._heldout_problems: Optional[List[Dict[str, str]]] = None

    def _setup_verifier(self):
        verifier_kwargs: dict[str, Any] = {}
        if self.config.verifier_type == "code":
            verifier_kwargs = {
                "timeout": self.config.verifier_timeout,
                "docker_image": self.config.code_docker_image,
                "memory_limit": self.config.code_memory_limit,
            }
        self.verifier = create_verifier(self.config.verifier_type, **verifier_kwargs)

    def setup(self):
        """Load tokenizer, policy model, reference model, and verifier."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._setup_verifier()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs, self.use_kbit_training = build_causal_lm_load_kwargs(
            prefer_bf16=self.config.bf16,
            allow_8bit=self.config.use_lora,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        if not torch.cuda.is_available():
            self.model.to(self.device)

        if self.config.use_lora:
            self._apply_lora()
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        ref_model_kwargs, _ = build_causal_lm_load_kwargs(
            prefer_bf16=self.config.bf16,
            allow_8bit=True,
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **ref_model_kwargs,
        )
        if not torch.cuda.is_available():
            self.ref_model.to(self.device)
        self.ref_model.eval()

        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate)
        logger.info("Loaded GRPO trainer components for %s", self.config.model_name)

    def _apply_lora(self):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if self.use_kbit_training:
            self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        self.model.print_trainable_parameters()

    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)

        gathered = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        return (gathered * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    def compute_grpo_loss(
        self,
        prompt: str,
        chosen_responses: List[str],
        rejected_responses: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss and monitoring metrics for a grouped prompt."""
        device = next(self.model.parameters()).device
        n_each = max(self.config.group_size // 2, 1)

        responses: list[str] = []
        raw_rewards: list[float] = []
        advantages: list[float] = []

        for response in chosen_responses[:n_each]:
            responses.append(response)
            raw_rewards.append(1.0)
            advantages.append(1.0)

        for response in rejected_responses[:n_each]:
            responses.append(response)
            raw_rewards.append(0.0)
            advantages.append(-1.0)

        if not responses:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {
                "policy_loss": 0.0,
                "kl_div": 0.0,
                "ratio_mean": 1.0,
                "mean_reward": 0.0,
                "reward_std": 0.0,
            }

        advantages_tensor = torch.tensor(advantages, device=device, dtype=torch.float32)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        target_tensor = torch.tensor(raw_rewards, device=device, dtype=torch.float32)

        sequences = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for response in responses
        ]
        encodings = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        policy_log_probs = self.compute_log_probs(self.model, input_ids, attention_mask, input_ids)
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(self.ref_model, input_ids, attention_mask, input_ids)

        log_ratio = policy_log_probs - ref_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
        policy_loss = -torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor).mean()

        # Positive KL proxy that is stable for per-sample logging.
        kl_div = torch.mean(torch.exp(log_ratio) - 1.0 - log_ratio)
        loss = policy_loss + self.config.kl_coef * kl_div

        preference_confidence = torch.where(
            target_tensor > 0.5,
            torch.sigmoid(log_ratio),
            1.0 - torch.sigmoid(log_ratio),
        )

        metrics = {
            "policy_loss": float(policy_loss.item()),
            "kl_div": float(kl_div.item()),
            "ratio_mean": float(ratio.mean().item()),
            "mean_reward": float(preference_confidence.mean().item()),
            "reward_std": float(preference_confidence.std(unbiased=False).item()),
        }
        return loss, metrics

    def _build_scheduler(self, dataset_size: int):
        from transformers import get_linear_schedule_with_warmup

        steps_per_epoch = math.ceil(dataset_size / max(self.config.batch_size, 1))
        optimizer_steps = math.ceil(steps_per_epoch / max(self.config.gradient_accumulation_steps, 1))
        total_steps = max(optimizer_steps * self.config.num_epochs, 1)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return total_steps

    def _load_heldout_problems(self) -> List[Dict[str, str]]:
        if self._heldout_problems is not None:
            return self._heldout_problems

        try:
            from data_generator.dataset_loader import GSM8KLoader

            loader = GSM8KLoader(
                split=self.config.heldout_split,
                subset_size=self.config.heldout_eval_size,
            )
            problems = loader.load()
            self._heldout_problems = [
                {"prompt": problem.problem, "answer": problem.answer}
                for problem in problems
            ]
        except Exception as exc:
            logger.warning("Unable to load held-out GSM8K problems: %s", exc)
            self._heldout_problems = []

        return self._heldout_problems

    def _generate_greedy_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.eval_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)

    def evaluate_checkpoint(self, step: int) -> Optional[float]:
        """
        Evaluate pass@1 on held-out GSM8K problems every fixed number of optimizer steps.
        """
        if self.config.verifier_type != "math":
            logger.info("Skipping held-out checkpoint eval because verifier_type=%s", self.config.verifier_type)
            return None

        problems = self._load_heldout_problems()
        if not problems:
            return None

        self.model.eval()
        correct = 0
        rewards: list[float] = []

        for problem in problems:
            response = self._generate_greedy_response(problem["prompt"])
            result = self.verifier.verify_reasoning_path(response, problem["answer"])
            is_correct = result.status.value == "correct"
            reward = 1.0 if is_correct else 0.0
            rewards.append(reward)
            if is_correct:
                correct += 1

        pass_at_1 = correct / len(problems)
        log_to_wandb(
            {
                f"eval/pass_at_1_step_{step}": pass_at_1,
                "eval/mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            },
            step=step,
        )
        self.model.train()
        return pass_at_1

    def train(self, data: List[Dict[str, Any]]):
        """Train the policy with custom GRPO updates."""
        self.setup()

        dataset = GRPODataset(
            data,
            self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        if len(dataset) == 0:
            logger.warning("No GRPO groups available for training.")
            return

        total_optimizer_steps = self._build_scheduler(len(dataset))
        ensure_wandb_run(
            project=self.config.wandb_project,
            name="grpo-training",
            mode=self.config.wandb_mode,
            config={
                **asdict(self.config),
                "dataset_size": len(dataset),
                "reward_type": self.config.verifier_type,
                "training_steps": total_optimizer_steps,
            },
            tags=["grpo", self.config.verifier_type],
        )

        self.model.train()
        self.optimizer.zero_grad()
        global_step = 0
        total_loss = 0.0

        log_dir = Path(self.config.output_dir) / "training_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = log_dir / "training_metrics.jsonl"

        from tqdm import tqdm

        for epoch in range(self.config.num_epochs):
            logger.info("GRPO epoch %s/%s", epoch + 1, self.config.num_epochs)
            epoch_loss = 0.0
            progress = tqdm(range(0, len(dataset), self.config.batch_size), desc=f"Epoch {epoch + 1}")
            micro_batches_since_update = 0

            for batch_idx in progress:
                accumulated_loss = 0.0
                batch_metrics = {
                    "policy_loss": 0.0,
                    "kl_div": 0.0,
                    "ratio_mean": 0.0,
                    "mean_reward": 0.0,
                    "reward_std": 0.0,
                }
                items_in_batch = min(self.config.batch_size, len(dataset) - batch_idx)

                for item_offset in range(items_in_batch):
                    group = dataset[batch_idx + item_offset]
                    loss, metrics = self.compute_grpo_loss(
                        group["prompt"],
                        group["chosen"],
                        group["rejected"],
                    )
                    (loss / self.config.gradient_accumulation_steps).backward()
                    accumulated_loss += float(loss.item())
                    for key, value in metrics.items():
                        batch_metrics[key] += value

                epoch_loss += accumulated_loss
                progress.set_postfix({"loss": f"{accumulated_loss / max(items_in_batch, 1):.4f}"})
                micro_batches_since_update += 1

                is_last_batch = batch_idx + items_in_batch >= len(dataset)
                if micro_batches_since_update < self.config.gradient_accumulation_steps and not is_last_batch:
                    total_loss += accumulated_loss
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                avg_kl = batch_metrics["kl_div"] / max(items_in_batch, 1)
                lr_reduced = avg_kl > self.config.kl_threshold

                if lr_reduced:
                    logger.warning(
                        "KL divergence %.4f exceeded threshold %.4f; reducing LR by 10%% for this step.",
                        avg_kl,
                        self.config.kl_threshold,
                    )
                    for group in self.optimizer.param_groups:
                        group["lr"] *= 0.9

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                global_step += 1
                micro_batches_since_update = 0

                current_lr = self.scheduler.get_last_lr()[0]
                mean_reward = batch_metrics["mean_reward"] / max(items_in_batch, 1)
                reward_std = batch_metrics["reward_std"] / max(items_in_batch, 1)
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + (batch_idx / len(dataset)),
                    "loss": accumulated_loss / max(items_in_batch, 1),
                    "policy_loss": batch_metrics["policy_loss"] / max(items_in_batch, 1),
                    "kl_divergence": avg_kl,
                    "gradient_norm": float(grad_norm),
                    "ratio_mean": batch_metrics["ratio_mean"] / max(items_in_batch, 1),
                    "learning_rate": current_lr,
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "lr_reduced_for_kl": lr_reduced,
                }

                with open(metrics_file, "a") as handle:
                    handle.write(json.dumps(log_entry) + "\n")

                log_to_wandb(
                    {
                        "train/loss": log_entry["loss"],
                        "train/policy_loss": log_entry["policy_loss"],
                        "train/kl_divergence": avg_kl,
                        "train/gradient_norm": float(grad_norm),
                        "train/ratio_mean": log_entry["ratio_mean"],
                        "train/learning_rate": current_lr,
                        "train/mean_reward": mean_reward,
                        "train/reward_std": reward_std,
                    },
                    step=global_step,
                )

                if self.config.eval_interval_steps > 0 and global_step % self.config.eval_interval_steps == 0:
                    self.evaluate_checkpoint(global_step)

                total_loss += accumulated_loss

            logger.info(
                "Epoch %s complete. Average loss: %.4f",
                epoch + 1,
                epoch_loss / max(len(dataset), 1),
            )

        self.save()
        avg_loss = total_loss / max(len(dataset), 1)
        logger.info("GRPO training complete. Average loss: %.4f", avg_loss)

    def save(self, path: Optional[str] = None):
        save_path = Path(path or self.config.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.config.use_lora and hasattr(self.model, "merge_and_unload"):
            logger.info("Merging LoRA adapters before save")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)

        self.tokenizer.save_pretrained(save_path)
        logger.info("Saved GRPO model to %s", save_path)


def train_grpo_from_synthetic_data(
    data_path: str,
    output_dir: str = "./grpo_output",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_epochs: int = 1,
    batch_size: int = 2,
    verifier_type: str = "math",
):
    """Convenience entry point for GRPO training from a JSONL file."""
    data = []
    with open(data_path) as handle:
        for line in handle:
            data.append(json.loads(line))

    config = GRPOConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        verifier_type=verifier_type,
    )

    trainer = ReasoningGRPOTrainer(config)
    trainer.train(data)
    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model with GRPO from synthetic data")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./grpo_output")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--verifier-type", type=str, default="math", choices=["math", "code"])
    args = parser.parse_args()

    train_grpo_from_synthetic_data(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        verifier_type=args.verifier_type,
    )
