"""
Group Relative Policy Optimization (GRPO) trainer with verifier-aware logging.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

# Ensure sibling src modules are importable when this file runs as a script.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from verifier import create_verifier, get_default_sandbox_image
from wandb_utils import ensure_wandb_run, log_to_wandb
from prompting import format_conversation, format_prompt

try:
    from .runtime_utils import build_causal_lm_load_kwargs, get_runtime_device
except ImportError:
    from training.runtime_utils import build_causal_lm_load_kwargs, get_runtime_device

logger = logging.getLogger(__name__)


def _resolve_distributed_timeout(timeout_minutes: int) -> timedelta:
    """
    Map user-facing timeout semantics to a concrete process-group timeout.

    PyTorch/NCCL expects a timeout value for collectives. We treat non-positive
    values as "no practical timeout" and map them to a very large timeout
    instead of inheriting NCCL's default 10-minute watchdog.
    """
    if timeout_minutes <= 0:
        return timedelta(days=365)
    return timedelta(minutes=timeout_minutes)


def _parse_requested_num_gpus(value: str, available: int) -> int:
    """Resolve requested GPU count from a user value and available hardware."""
    normalized = str(value).strip().lower()
    if normalized in {"auto", "all"}:
        return max(1, available)

    try:
        requested = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid --num-gpus value: {value}") from exc

    if requested < 1:
        raise ValueError("--num-gpus must be >= 1")
    return min(requested, max(1, available))


def maybe_launch_grpo_distributed(training_args: List[str], requested_num_gpus: str = "auto") -> bool:
    """
    Launch GRPO training via torch.distributed.run when multiple GPUs are available.

    Returns True if this process launched a distributed child and should exit.
    """
    if os.environ.get("LOCAL_RANK") is not None or os.environ.get("RANK") is not None:
        return False

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    target_gpus = _parse_requested_num_gpus(requested_num_gpus, available_gpus)
    if target_gpus <= 1:
        logger.info(
            "Distributed launch not required (requested=%s, available=%s). Running single-process GRPO.",
            requested_num_gpus,
            available_gpus,
        )
        return False

    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(target_gpus),
        str(script_path),
        *training_args,
    ]
    logger.info("Launching distributed GRPO across %s GPUs", target_gpus)
    subprocess.run(cmd, check=True)
    return True


class RayMathVerificationPool:
    """
    Local Ray actor pool for parallel math verification.
    Works in single-node environments like Kaggle notebooks.
    """

    def __init__(self, num_workers: int = 4):
        self.num_workers = max(1, num_workers)
        self._ray = None
        self._actors = []
        self._next_actor = 0
        self._enabled = False
        self._stats: Dict[str, int] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
        }
        self._local_verifier = create_verifier("math")
        self._init_pool()

    def _init_pool(self):
        try:
            import ray

            self._ray = ray
            ray_pythonpath = os.pathsep.join(
                [str(_SRC_ROOT), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    log_to_driver=False,
                    namespace="grpo_verification",
                    runtime_env={"env_vars": {"PYTHONPATH": ray_pythonpath}},
                )

            @ray.remote
            class _MathVerifierWorker:
                def __init__(self):
                    import sys

                    if str(_SRC_ROOT) not in sys.path:
                        sys.path.insert(0, str(_SRC_ROOT))

                    from verifier import create_verifier as create_remote_verifier

                    self.verifier = create_remote_verifier("math")

                def verify(self, reasoning: str, expected_answer: str) -> str:
                    result = self.verifier.verify_reasoning_path(reasoning, expected_answer)
                    return result.status.value

            self._actors = [_MathVerifierWorker.remote() for _ in range(self.num_workers)]
            self._enabled = len(self._actors) > 0
            logger.info("Initialized Ray math verification pool with %s workers", len(self._actors))
        except Exception as exc:
            logger.warning("Ray verification pool unavailable, using local verification fallback: %s", exc)
            self._enabled = False
            self._actors = []

    def verify_batch(self, responses: List[str], expected_answer: str) -> List[str]:
        if not responses:
            return []
        ticket = self.submit_verify_batch(responses, expected_answer)
        return self.resolve_verify_batch(ticket)

    def submit_verify_batch(self, responses: List[str], expected_answer: str):
        if not responses:
            return ("empty", [])

        self._stats["submitted"] += len(responses)

        if not self._enabled:
            statuses = []
            for response in responses:
                result = self._local_verifier.verify_reasoning_path(response, expected_answer)
                statuses.append(result.status.value)
            return ("local", statuses)

        futures = []
        for response in responses:
            actor = self._actors[self._next_actor]
            self._next_actor = (self._next_actor + 1) % len(self._actors)
            futures.append(actor.verify.remote(response, expected_answer))
        return ("ray", futures, responses, expected_answer)

    def resolve_verify_batch(self, ticket) -> List[str]:
        mode = ticket[0]
        if mode == "empty":
            return []
        if mode == "local":
            statuses = ticket[1]
            self._stats["completed"] += len(statuses)
            return statuses

        _, futures, responses, expected_answer = ticket
        try:
            statuses = self._ray.get(futures)
            self._stats["completed"] += len(statuses)
            return statuses
        except Exception as exc:
            logger.warning("Ray verify_batch failed, falling back to local verifier: %s", exc)
            self._stats["failed"] += len(responses)
            statuses = []
            for response in responses:
                result = self._local_verifier.verify_reasoning_path(response, expected_answer)
                statuses.append(result.status.value)
            self._stats["completed"] += len(statuses)
            return statuses

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    group_size: int = 8
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
    prompt_problem_type: str = "math"

    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    logging_steps: int = 10
    eval_interval_steps: int = 50
    heldout_eval_size: int = 20
    heldout_split: str = "test"
    heldout_dataset: str = "gsm8k"
    eval_max_new_tokens: int = 256
    output_dir: str = "./grpo_output"
    save_best_checkpoint: bool = True
    best_checkpoint_metric: str = "pass_at_1"
    min_eval_improvement: float = 1e-4
    early_stop_patience: int = 0

    bf16: bool = True
    gradient_checkpointing: bool = True
    distributed_timeout_minutes: int = 0

    verifier_type: str = "math"
    verifier_timeout: int = 10
    code_docker_image: str = field(default_factory=get_default_sandbox_image)
    code_memory_limit: str = "512m"

    online_max_new_tokens: int = 256
    online_temperature: float = 0.8
    online_top_p: float = 0.95
    online_resample_attempts: int = 2
    online_min_reward_std: float = 1e-6
    enable_ray_verification: bool = True
    ray_verifier_workers: int = 4

    reward_correct: float = 1.0
    reward_incorrect: float = 0.0
    reward_parse_error: float = -0.25
    reward_timeout: float = -0.25
    reward_unknown: float = -0.1
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0

    wandb_project: str = "distributed-reasoning-loop"
    wandb_mode: str = "offline"


SUPPORTED_CHECKPOINT_METRICS = {
    "pass_at_1",
    "mean_reward",
}


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
                    {"prompt": prompt, "chosen": [], "rejected": [], "expected_answer": None},
                )
                bucket["chosen"].append(item["chosen"])
                bucket["rejected"].append(item["rejected"])
                if bucket["expected_answer"] is None:
                    bucket["expected_answer"] = (
                        item.get("expected_answer")
                        or item.get("chosen_answer")
                        or item.get("answer")
                    )
                continue

            prompt = item.get("prompt", item.get("problem", ""))
            bucket = prompt_to_responses.setdefault(
                prompt,
                {"prompt": prompt, "chosen": [], "rejected": [], "expected_answer": None},
            )

            response = item.get("reasoning", item.get("response", ""))
            if item.get("is_correct", False):
                bucket["chosen"].append(response)
            else:
                bucket["rejected"].append(response)

            if bucket["expected_answer"] is None:
                bucket["expected_answer"] = item.get("expected_answer") or item.get("answer")

        groups = [
            group
            for group in prompt_to_responses.values()
            if group["prompt"] and (group["expected_answer"] is not None or len(group["chosen"]) > 0)
        ]
        logger.info("Created %s GRPO prompt groups", len(groups))
        return groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.groups[idx]


class ReasoningGRPOTrainer:
    """Custom GRPO trainer with online verifier-backed rewards."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        if self.config.best_checkpoint_metric not in SUPPORTED_CHECKPOINT_METRICS:
            raise ValueError(
                "Unsupported best_checkpoint_metric="
                f"{self.config.best_checkpoint_metric!r}. "
                f"Expected one of {sorted(SUPPORTED_CHECKPOINT_METRICS)}."
            )
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.device = get_runtime_device()
        self.use_kbit_training = False
        self.verifier = None
        self.verification_pool: Optional[RayMathVerificationPool] = None
        self._heldout_problems: Optional[List[Dict[str, str]]] = None
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        self.best_eval_score: Optional[float] = None
        self.best_eval_step: Optional[int] = None
        self.no_improvement_evals: int = 0

    def _is_main_process(self) -> bool:
        return self.rank == 0

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _setup_distributed(self):
        if not self.distributed:
            return

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            backend = "nccl"
        else:
            self.device = torch.device("cpu")
            backend = "gloo"

        effective_timeout = _resolve_distributed_timeout(self.config.distributed_timeout_minutes)
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                timeout=effective_timeout,
            )
        logger.info(
            "Initialized distributed GRPO rank=%s local_rank=%s world_size=%s backend=%s timeout_minutes=%s effective_timeout=%s",
            self.rank,
            self.local_rank,
            self.world_size,
            backend,
            self.config.distributed_timeout_minutes,
            effective_timeout,
        )

    def _setup_verifier(self):
        verifier_kwargs: dict[str, Any] = {}
        if self.config.verifier_type == "code":
            verifier_kwargs = {
                "timeout": self.config.verifier_timeout,
                "docker_image": self.config.code_docker_image,
                "memory_limit": self.config.code_memory_limit,
            }
        self.verifier = create_verifier(self.config.verifier_type, **verifier_kwargs)
        if self.config.verifier_type == "math" and self.config.enable_ray_verification:
            per_rank_workers = max(1, self.config.ray_verifier_workers // max(self.world_size, 1))
            self.verification_pool = RayMathVerificationPool(
                num_workers=per_rank_workers
            )

    def _build_sequence_encodings(self, prompt: str, responses: List[str], device: torch.device):
        sequences = [
            format_conversation(
                self.tokenizer,
                prompt,
                response,
                problem_type=self.config.prompt_problem_type,
            )
            for response in responses
        ]
        return self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(device)

    def _status_to_reward(self, status: str) -> float:
        if status == "correct":
            reward = self.config.reward_correct
        elif status == "incorrect":
            reward = self.config.reward_incorrect
        elif status == "parse_error":
            reward = self.config.reward_parse_error
        elif status == "timeout":
            reward = self.config.reward_timeout
        else:
            reward = self.config.reward_unknown

        return float(
            max(
                self.config.reward_clip_min,
                min(self.config.reward_clip_max, reward),
            )
        )

    def _score_response(self, response: str, expected_answer: Optional[str]) -> float:
        if self.config.verifier_type != "math" or not expected_answer:
            return 0.0

        result = self.verifier.verify_reasoning_path(response, expected_answer)
        return self._status_to_reward(result.status.value)

    def _resolve_expected_answer(self, group: Dict[str, Any]) -> Optional[str]:
        expected = group.get("expected_answer")
        if expected:
            return expected

        if self.config.verifier_type != "math":
            return None

        if not hasattr(self.verifier, "extract_final_answer"):
            return None

        # Fall back to extracting from a known-good chosen response.
        for chosen in group.get("chosen", []):
            answer = self.verifier.extract_final_answer(chosen)
            if answer:
                return answer
        return None

    def _sample_group_responses(self, prompt: str) -> List[str]:
        prompt_text = format_prompt(
            self.tokenizer,
            prompt,
            problem_type=self.config.prompt_problem_type,
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        policy_model = self._unwrap_model()
        device = next(policy_model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        input_ids = inputs["input_ids"].repeat(self.config.group_size, 1)
        attention_mask = inputs["attention_mask"].repeat(self.config.group_size, 1)
        prompt_len = inputs["input_ids"].shape[1]

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.online_max_new_tokens,
                do_sample=True,
                temperature=self.config.online_temperature,
                top_p=self.config.online_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if was_training:
            self.model.train()

        responses: List[str] = []
        for idx in range(outputs.shape[0]):
            responses.append(
                self.tokenizer.decode(
                    outputs[idx, prompt_len:],
                    skip_special_tokens=True,
                )
            )
        return responses

    def setup(self):
        """Load tokenizer, policy model, reference model, and verifier."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._setup_distributed()
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
        if self.distributed and torch.cuda.is_available():
            model_kwargs.pop("device_map", None)
            model_kwargs["device_map"] = {"": self.local_rank}
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        if torch.cuda.is_available():
            if self.distributed:
                self.model.to(self.device)
        else:
            self.model.to(self.device)

        if self.config.use_lora:
            self._apply_lora()
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.config.use_lora:
            self.ref_model = None
            logger.info("Using adapter-disabled policy model as GRPO reference model to reduce memory.")
        else:
            ref_model_kwargs, _ = build_causal_lm_load_kwargs(
                prefer_bf16=self.config.bf16,
                allow_8bit=True,
            )
            if self.distributed and torch.cuda.is_available():
                ref_model_kwargs.pop("device_map", None)
                ref_model_kwargs["device_map"] = {"": self.local_rank}
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **ref_model_kwargs,
            )
            if torch.cuda.is_available():
                if self.distributed:
                    self.ref_model.to(self.device)
            else:
                self.ref_model.to(self.device)
            self.ref_model.eval()

        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
            )

        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate)
        if self._is_main_process():
            logger.info(
                "Loaded GRPO trainer components for %s (distributed=%s, world_size=%s)",
                self.config.model_name,
                self.distributed,
                self.world_size,
            )

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
        selected_logits = torch.gather(
            shift_logits,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        token_log_probs = selected_logits - torch.logsumexp(shift_logits, dim=-1)

        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    def _adapter_disabled_context(self, model):
        disable_adapter = getattr(model, "disable_adapter", None)
        if callable(disable_adapter):
            return disable_adapter()
        return nullcontext()

    def _zero_loss_metrics(
        self,
        device: torch.device,
        mean_reward: float = 0.0,
        reward_std: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "policy_loss": 0.0,
            "kl_div": 0.0,
            "ratio_mean": 1.0,
            "mean_reward": float(mean_reward),
            "reward_std": float(reward_std),
        }

    def _verify_responses(self, responses: List[str], expected_answer: str) -> List[str]:
        if self.verification_pool is not None:
            return self.verification_pool.verify_batch(responses, expected_answer)
        statuses = []
        for response in responses:
            result = self.verifier.verify_reasoning_path(response, expected_answer)
            statuses.append(result.status.value)
        return statuses

    def _reward_stats_from_statuses(self, statuses: List[str]) -> Tuple[torch.Tensor, float]:
        rewards = [self._status_to_reward(status) for status in statuses]
        rewards_tensor = torch.tensor(
            rewards,
            device=next(self.model.parameters()).device,
            dtype=torch.float32,
        )
        reward_std = float(rewards_tensor.std(unbiased=False).item()) if len(rewards) > 0 else 0.0
        return rewards_tensor, reward_std

    def _compute_loss_from_verified_statuses(
        self,
        prompt: str,
        responses: List[str],
        statuses: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = next(self.model.parameters()).device
        if not responses or not statuses:
            return self._zero_loss_metrics(device)

        rewards_tensor, reward_std = self._reward_stats_from_statuses(statuses)
        if reward_std < self.config.online_min_reward_std:
            return self._zero_loss_metrics(
                device,
                mean_reward=float(rewards_tensor.mean().item()) if rewards_tensor.numel() > 0 else 0.0,
                reward_std=reward_std,
            )

        if reward_std < 1e-8:
            advantages_tensor = rewards_tensor - rewards_tensor.mean()
        else:
            advantages_tensor = (rewards_tensor - rewards_tensor.mean()) / (reward_std + 1e-8)

        encodings = self._build_sequence_encodings(prompt, responses, device)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.no_grad():
            old_log_probs = self.compute_log_probs(self.model, input_ids, attention_mask, input_ids)
        policy_log_probs = self.compute_log_probs(self.model, input_ids, attention_mask, input_ids)

        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
        policy_loss = -torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor).mean()

        with torch.no_grad():
            if self.ref_model is not None:
                ref_log_probs = self.compute_log_probs(self.ref_model, input_ids, attention_mask, input_ids)
            else:
                reference_model = self._unwrap_model()
                with self._adapter_disabled_context(reference_model):
                    ref_log_probs = self.compute_log_probs(
                        reference_model,
                        input_ids,
                        attention_mask,
                        input_ids,
                    )
        log_ratio_ref = policy_log_probs - ref_log_probs
        kl_div = torch.mean(torch.exp(log_ratio_ref) - 1.0 - log_ratio_ref)
        loss = policy_loss + self.config.kl_coef * kl_div

        return loss, {
            "policy_loss": float(policy_loss.item()),
            "kl_div": float(kl_div.item()),
            "ratio_mean": float(ratio.mean().item()),
            "mean_reward": float(rewards_tensor.mean().item()),
            "reward_std": reward_std,
        }

    def compute_online_grpo_loss(
        self,
        prompt: str,
        expected_answer: Optional[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss from online generation and verifier-derived rewards.
        """
        device = next(self.model.parameters()).device
        if not expected_answer:
            return self._zero_loss_metrics(device)

        responses: List[str] = []
        statuses: List[str] = []
        reward_std = 0.0

        # Resample a few times to reduce zero-variance groups (all-correct/all-incorrect),
        # which otherwise produce near-zero advantages and weak gradient signal.
        max_attempts = max(1, self.config.online_resample_attempts + 1)
        for _ in range(max_attempts):
            responses = self._sample_group_responses(prompt)
            if not responses:
                break
            statuses = self._verify_responses(responses, expected_answer)
            _, reward_std = self._reward_stats_from_statuses(statuses)
            if reward_std >= self.config.online_min_reward_std:
                break

        if not responses:
            return self._zero_loss_metrics(device)

        return self._compute_loss_from_verified_statuses(prompt, responses, statuses)

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
            from data_generator.dataset_loader import get_loader

            loader = get_loader(
                self.config.heldout_dataset,
                split=self.config.heldout_split,
                subset_size=self.config.heldout_eval_size,
            )
            problems = loader.load()
            self._heldout_problems = [
                {"prompt": problem.problem, "answer": problem.answer}
                for problem in problems
            ]
        except Exception as exc:
            logger.warning(
                "Unable to load held-out %s problems: %s",
                self.config.heldout_dataset,
                exc,
            )
            self._heldout_problems = []

        return self._heldout_problems

    def _generate_greedy_response(self, prompt: str) -> str:
        prompt_text = format_prompt(
            self.tokenizer,
            prompt,
            problem_type=self.config.prompt_problem_type,
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )

        policy_model = self._unwrap_model()
        device = next(policy_model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = policy_model.generate(
                **inputs,
                max_new_tokens=self.config.eval_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)

    def evaluate_checkpoint(self, step: int) -> Optional[Dict[str, float]]:
        """
        Evaluate pass@1 on held-out GSM8K problems every fixed number of optimizer steps.
        """
        if self.config.verifier_type != "math":
            logger.info("Skipping held-out checkpoint eval because verifier_type=%s", self.config.verifier_type)
            return None
        if self.distributed and not self._is_main_process():
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
        metrics = {
            "step": step,
            "dataset": self.config.heldout_dataset,
            "split": self.config.heldout_split,
            "pass_at_1": pass_at_1,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "num_problems": len(problems),
            "num_correct": correct,
        }
        log_to_wandb(
            {
                f"eval/pass_at_1_step_{step}": pass_at_1,
                "eval/mean_reward": metrics["mean_reward"],
            },
            step=step,
        )
        eval_dir = Path(self.config.output_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "heldout_eval_history.jsonl", "a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        self.model.train()
        return metrics

    def _selection_metric_value(self, metrics: Dict[str, float]) -> Optional[float]:
        value = metrics.get(self.config.best_checkpoint_metric)
        if value is None:
            logger.warning(
                "Held-out metrics are missing %s; skipping best-checkpoint update.",
                self.config.best_checkpoint_metric,
            )
            return None
        return float(value)

    def _write_selection_state(self) -> None:
        if not self._is_main_process():
            return
        selection_state = {
            "best_eval_score": self.best_eval_score,
            "best_eval_step": self.best_eval_step,
            "no_improvement_evals": self.no_improvement_evals,
            "heldout_dataset": self.config.heldout_dataset,
            "heldout_split": self.config.heldout_split,
            "best_checkpoint_metric": self.config.best_checkpoint_metric,
        }
        selection_dir = Path(self.config.output_dir)
        selection_dir.mkdir(parents=True, exist_ok=True)
        with open(selection_dir / "checkpoint_selection.json", "w") as handle:
            json.dump(selection_state, handle, indent=2)

    def _maybe_update_best_checkpoint(
        self,
        step: int,
        metrics: Optional[Dict[str, float]],
    ) -> bool:
        if metrics is None or not self._is_main_process():
            return False

        score = self._selection_metric_value(metrics)
        if score is None:
            return False

        is_improved = (
            self.best_eval_score is None
            or score > self.best_eval_score + self.config.min_eval_improvement
        )
        if is_improved:
            self.best_eval_score = score
            self.best_eval_step = step
            self.no_improvement_evals = 0
            if self.config.save_best_checkpoint:
                best_dir = Path(self.config.output_dir) / "best_checkpoint"
                self.save(best_dir, merge_lora=False)
            self._write_selection_state()
            logger.info(
                "New best GRPO checkpoint at step %s with held-out %s %.4f",
                step,
                self.config.best_checkpoint_metric,
                score,
            )
            return False

        self.no_improvement_evals += 1
        self._write_selection_state()
        logger.info(
            "Held-out %s %.4f did not improve best %.4f (patience %s/%s)",
            self.config.best_checkpoint_metric,
            score,
            self.best_eval_score,
            self.no_improvement_evals,
            self.config.early_stop_patience,
        )
        return (
            self.config.early_stop_patience > 0
            and self.no_improvement_evals >= self.config.early_stop_patience
        )

    def train(self, data: List[Dict[str, Any]]):
        """Train the policy with custom GRPO updates."""
        self.setup()
        if self.config.verifier_type != "math":
            raise ValueError("GRPO training currently supports verifier_type='math' only.")

        dataset = GRPODataset(
            data,
            self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        if len(dataset) == 0:
            logger.warning("No GRPO groups available for training.")
            return

        all_batch_starts = list(range(0, len(dataset), self.config.batch_size))
        if self.distributed:
            usable_batches = (len(all_batch_starts) // self.world_size) * self.world_size
            all_batch_starts = all_batch_starts[:usable_batches]
            if not all_batch_starts:
                if self._is_main_process():
                    logger.warning(
                        "Dataset too small for distributed run (groups=%s, batch_size=%s, world_size=%s).",
                        len(dataset),
                        self.config.batch_size,
                        self.world_size,
                    )
                return
            local_batch_starts = all_batch_starts[self.rank::self.world_size]
        else:
            local_batch_starts = all_batch_starts

        local_effective_dataset_size = max(1, len(local_batch_starts) * self.config.batch_size)
        total_optimizer_steps = self._build_scheduler(local_effective_dataset_size)
        if self._is_main_process():
            ensure_wandb_run(
                project=self.config.wandb_project,
                name="grpo-training",
                mode=self.config.wandb_mode,
                config={
                    **asdict(self.config),
                    "dataset_size": len(dataset),
                    "reward_type": self.config.verifier_type,
                    "training_steps": total_optimizer_steps,
                    "distributed": self.distributed,
                    "world_size": self.world_size,
                },
                tags=["grpo", self.config.verifier_type],
            )

        self.model.train()
        self.optimizer.zero_grad()
        global_step = 0
        total_loss = 0.0

        log_dir = Path(self.config.output_dir) / "training_logs"
        if self._is_main_process():
            log_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = log_dir / "training_metrics.jsonl"

        from tqdm import tqdm

        for epoch in range(self.config.num_epochs):
            if self._is_main_process():
                logger.info("GRPO epoch %s/%s", epoch + 1, self.config.num_epochs)
            epoch_loss = 0.0
            should_stop = False
            progress = tqdm(
                local_batch_starts,
                desc=f"Epoch {epoch + 1}",
                disable=not self._is_main_process(),
            )
            micro_batches_since_update = 0

            for local_batch_index, batch_idx in enumerate(progress):
                batch_wall_start = time.perf_counter()
                accumulated_loss = 0.0
                batch_metrics = {
                    "policy_loss": 0.0,
                    "kl_div": 0.0,
                    "ratio_mean": 0.0,
                    "mean_reward": 0.0,
                    "reward_std": 0.0,
                }
                rollout_time_s = 0.0
                verify_wait_time_s = 0.0
                resample_time_s = 0.0
                items_in_batch = min(self.config.batch_size, len(dataset) - batch_idx)

                pending_rollouts: List[Dict[str, Any]] = []
                for item_offset in range(items_in_batch):
                    group = dataset[batch_idx + item_offset]
                    prompt = group["prompt"]
                    expected_answer = self._resolve_expected_answer(group)

                    # Phase 1: rollout generation + async verification submit.
                    if not expected_answer:
                        pending_rollouts.append(
                            {
                                "prompt": prompt,
                                "expected_answer": None,
                                "responses": [],
                                "ticket": None,
                            }
                        )
                        continue

                    rollout_start = time.perf_counter()
                    responses = self._sample_group_responses(prompt)
                    rollout_time_s += time.perf_counter() - rollout_start
                    if not responses:
                        pending_rollouts.append(
                            {
                                "prompt": prompt,
                                "expected_answer": expected_answer,
                                "responses": [],
                                "ticket": None,
                            }
                        )
                        continue

                    if self.verification_pool is not None:
                        ticket = self.verification_pool.submit_verify_batch(responses, expected_answer)
                    else:
                        verify_start = time.perf_counter()
                        ticket = ("local", self._verify_responses(responses, expected_answer))
                        verify_wait_time_s += time.perf_counter() - verify_start

                    pending_rollouts.append(
                        {
                            "prompt": prompt,
                            "expected_answer": expected_answer,
                            "responses": responses,
                            "ticket": ticket,
                        }
                    )

                # Phase 2: resolve verification + compute losses.
                for pending in pending_rollouts:
                    prompt = pending["prompt"]
                    expected_answer = pending["expected_answer"]
                    responses = pending["responses"]
                    ticket = pending["ticket"]

                    if not expected_answer or not responses or ticket is None:
                        loss, metrics = self._zero_loss_metrics(next(self.model.parameters()).device)
                    else:
                        if self.verification_pool is not None:
                            resolve_start = time.perf_counter()
                            statuses = self.verification_pool.resolve_verify_batch(ticket)
                            verify_wait_time_s += time.perf_counter() - resolve_start
                        else:
                            statuses = ticket[1]

                        # If rewards are near-constant, retry with fresh responses to recover signal.
                        _, reward_std = self._reward_stats_from_statuses(statuses)
                        retries = 0
                        while (
                            reward_std < self.config.online_min_reward_std
                            and retries < self.config.online_resample_attempts
                        ):
                            resample_rollout_start = time.perf_counter()
                            responses = self._sample_group_responses(prompt)
                            delta_rollout = time.perf_counter() - resample_rollout_start
                            rollout_time_s += delta_rollout
                            resample_time_s += delta_rollout
                            if not responses:
                                break
                            resample_verify_start = time.perf_counter()
                            statuses = self._verify_responses(responses, expected_answer)
                            delta_verify = time.perf_counter() - resample_verify_start
                            verify_wait_time_s += delta_verify
                            resample_time_s += delta_verify
                            _, reward_std = self._reward_stats_from_statuses(statuses)
                            retries += 1

                        loss, metrics = self._compute_loss_from_verified_statuses(
                            prompt,
                            responses,
                            statuses if responses else [],
                        )

                    (loss / self.config.gradient_accumulation_steps).backward()
                    accumulated_loss += float(loss.item())
                    for key, value in metrics.items():
                        batch_metrics[key] += value

                epoch_loss += accumulated_loss
                if self._is_main_process():
                    progress.set_postfix({"loss": f"{accumulated_loss / max(items_in_batch, 1):.4f}"})
                micro_batches_since_update += 1

                is_last_batch = local_batch_index + 1 >= len(local_batch_starts)
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
                batch_wall_time_s = time.perf_counter() - batch_wall_start
                rollout_time_ms = rollout_time_s * 1000.0
                verify_wait_time_ms = verify_wait_time_s * 1000.0
                batch_wall_time_ms = batch_wall_time_s * 1000.0
                gpu_idle_estimate_ms = min(verify_wait_time_ms, batch_wall_time_ms)
                overlap_efficiency = (
                    rollout_time_ms / max(rollout_time_ms + verify_wait_time_ms, 1e-6)
                )
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
                    "rollout_time_ms": rollout_time_ms,
                    "verify_wait_time_ms": verify_wait_time_ms,
                    "gpu_idle_estimate_ms": gpu_idle_estimate_ms,
                    "batch_wall_time_ms": batch_wall_time_ms,
                    "overlap_efficiency": overlap_efficiency,
                    "resample_time_ms": resample_time_s * 1000.0,
                }

                if self._is_main_process():
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
                            "perf/rollout_time_ms": rollout_time_ms,
                            "perf/verify_wait_time_ms": verify_wait_time_ms,
                            "perf/gpu_idle_estimate_ms": gpu_idle_estimate_ms,
                            "perf/batch_wall_time_ms": batch_wall_time_ms,
                            "perf/overlap_efficiency": overlap_efficiency,
                            "perf/resample_time_ms": resample_time_s * 1000.0,
                        },
                        step=global_step,
                    )
                    if self.verification_pool is not None:
                        verify_stats = self.verification_pool.get_stats()
                        log_to_wandb(
                            {
                                "verify/submitted": verify_stats.get("submitted", 0),
                                "verify/completed": verify_stats.get("completed", 0),
                                "verify/failed": verify_stats.get("failed", 0),
                            },
                            step=global_step,
                        )

                if (
                    self._is_main_process()
                    and self.config.eval_interval_steps > 0
                    and global_step % self.config.eval_interval_steps == 0
                ):
                    should_stop = self._maybe_update_best_checkpoint(
                        global_step,
                        self.evaluate_checkpoint(global_step),
                    )
                else:
                    should_stop = False

                if self.distributed:
                    stop_tensor = torch.tensor(
                        [1 if should_stop else 0],
                        device=next(self.model.parameters()).device,
                        dtype=torch.int32,
                    )
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())

                if should_stop:
                    if self._is_main_process():
                        logger.info("Early stopping GRPO after step %s", global_step)
                    break

                total_loss += accumulated_loss

            if should_stop:
                break

            if self._is_main_process():
                logger.info(
                    "Epoch %s complete. Average loss: %.4f",
                    epoch + 1,
                    epoch_loss / max(len(local_batch_starts), 1),
                )

        if self._is_main_process():
            self.save()
            self._write_selection_state()
            avg_loss = total_loss / max(len(local_batch_starts), 1)
            logger.info("GRPO training complete. Average loss: %.4f", avg_loss)

        if self.distributed and dist.is_initialized():
            dist.barrier()

    def save(self, path: Optional[str] = None, merge_lora: bool = True):
        save_path = Path(path or self.config.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_to_save = self._unwrap_model()

        if self.config.use_lora and merge_lora and hasattr(model_to_save, "merge_and_unload"):
            logger.info("Merging LoRA adapters before save")
            merged_model = model_to_save.merge_and_unload()
            merged_model.save_pretrained(save_path)
        else:
            model_to_save.save_pretrained(save_path)

        self.tokenizer.save_pretrained(save_path)
        logger.info("Saved GRPO model to %s", save_path)


def train_grpo_from_synthetic_data(
    data_path: str,
    output_dir: str = "./grpo_output",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_epochs: int = 1,
    batch_size: int = 2,
    group_size: int = 8,
    online_max_new_tokens: int = 256,
    online_temperature: float = 0.8,
    online_top_p: float = 0.95,
    online_resample_attempts: int = 2,
    enable_ray_verification: bool = True,
    ray_verifier_workers: int = 4,
    verifier_type: str = "math",
    heldout_dataset: str = "gsm8k",
    heldout_split: str = "test",
    heldout_eval_size: int = 20,
    save_best_checkpoint: bool = True,
    best_checkpoint_metric: str = "pass_at_1",
    min_eval_improvement: float = 1e-4,
    early_stop_patience: int = 0,
    distributed_timeout_minutes: int = 0,
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
        group_size=group_size,
        online_max_new_tokens=online_max_new_tokens,
        online_temperature=online_temperature,
        online_top_p=online_top_p,
        online_resample_attempts=online_resample_attempts,
        enable_ray_verification=enable_ray_verification,
        ray_verifier_workers=ray_verifier_workers,
        verifier_type=verifier_type,
        heldout_dataset=heldout_dataset,
        heldout_split=heldout_split,
        heldout_eval_size=heldout_eval_size,
        save_best_checkpoint=save_best_checkpoint,
        best_checkpoint_metric=best_checkpoint_metric,
        min_eval_improvement=min_eval_improvement,
        early_stop_patience=early_stop_patience,
        distributed_timeout_minutes=distributed_timeout_minutes,
    )

    trainer = ReasoningGRPOTrainer(config)
    trainer.train(data)
    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model with GRPO from synthetic data")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./grpo_output")
    parser.add_argument(
        "--model-name",
        "--model",
        dest="model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--num-epochs",
        "--epochs",
        dest="num_epochs",
        type=int,
        default=1,
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--online-max-new-tokens", type=int, default=256)
    parser.add_argument("--online-temperature", type=float, default=0.8)
    parser.add_argument("--online-top-p", type=float, default=0.95)
    parser.add_argument("--online-resample-attempts", type=int, default=2)
    parser.add_argument("--enable-ray-verification", action="store_true")
    parser.add_argument("--disable-ray-verification", action="store_true")
    parser.add_argument("--ray-verifier-workers", type=int, default=4)
    parser.add_argument("--verifier-type", type=str, default="math", choices=["math", "code"])
    parser.add_argument("--heldout-dataset", type=str, default="gsm8k")
    parser.add_argument("--heldout-split", type=str, default="test")
    parser.add_argument("--heldout-eval-size", type=int, default=20)
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--disable-save-best-checkpoint", action="store_true")
    parser.add_argument("--best-checkpoint-metric", type=str, default="pass_at_1")
    parser.add_argument("--min-eval-improvement", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=0,
        help="Distributed process-group timeout in minutes. Use 0 for no practical timeout.",
    )
    parser.add_argument(
        "--num-gpus",
        type=str,
        default="auto",
        help="Number of GPUs to use: auto|all|1|2|3|...",
    )
    args = parser.parse_args()

    original_args = sys.argv[1:]
    forwarded_args: List[str] = []
    skip_next = False
    for idx, token in enumerate(original_args):
        if skip_next:
            skip_next = False
            continue

        if token == "--num-gpus":
            next_token = original_args[idx + 1] if idx + 1 < len(original_args) else None
            if next_token and not next_token.startswith("--"):
                skip_next = True
            continue

        if token.startswith("--num-gpus="):
            continue

        forwarded_args.append(token)

    if maybe_launch_grpo_distributed(forwarded_args, requested_num_gpus=args.num_gpus):
        sys.exit(0)

    train_grpo_from_synthetic_data(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        group_size=args.group_size,
        online_max_new_tokens=args.online_max_new_tokens,
        online_temperature=args.online_temperature,
        online_top_p=args.online_top_p,
        online_resample_attempts=args.online_resample_attempts,
        enable_ray_verification=(False if args.disable_ray_verification else True if args.enable_ray_verification else True),
        ray_verifier_workers=args.ray_verifier_workers,
        verifier_type=args.verifier_type,
        heldout_dataset=args.heldout_dataset,
        heldout_split=args.heldout_split,
        heldout_eval_size=args.heldout_eval_size,
        save_best_checkpoint=(False if args.disable_save_best_checkpoint else True if args.save_best_checkpoint else True),
        best_checkpoint_metric=args.best_checkpoint_metric,
        min_eval_improvement=args.min_eval_improvement,
        early_stop_patience=args.early_stop_patience,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
    )
