"""
MoE pre-trainer.

Wraps :class:`MoEModel` with FSDP for the non-expert parameters (attention,
norms, embeddings, router gates) while keeping the expert FFNs partitioned
across ``expert_parallel_size`` ranks. The training loop logs per-step expert
utilization histograms and coefficient-of-variation so routing imbalance is
visible live.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from ..fsdp_utils import (
    FSDPConfig,
    accumulate_grad_steps,
    destroy_distributed,
    init_distributed,
    is_distributed,
    is_main_process,
    save_distributed_checkpoint,
    world_size,
    wrap_with_fsdp,
)
from ..precision import (
    PRECISION_BF16,
    PrecisionConfig,
    PrecisionPolicy,
    env_override_precision,
)
from ..pretraining.data import PretrainingDataConfig, build_pretraining_loader
from .model import MoEConfig, MoEModel

logger = logging.getLogger(__name__)


@dataclass
class MoEPreTrainerConfig:
    model: MoEConfig = field(default_factory=MoEConfig)
    data: PretrainingDataConfig = field(default_factory=PretrainingDataConfig)
    precision: str = PRECISION_BF16

    output_dir: str = "./outputs/moe_pretraining"
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    total_steps: int = 10_000
    log_every: int = 25
    checkpoint_every: int = 500
    seed: int = 17


class MoEPreTrainer:
    def __init__(self, config: MoEPreTrainerConfig):
        self.config = config
        self._setup_seed()
        init_distributed()
        self.world_size = world_size()
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = (
            torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.precision_policy = PrecisionPolicy(
            PrecisionConfig(
                precision=env_override_precision(default=config.precision),
                grad_clip=config.grad_clip,
            )
        )
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.loader = build_pretraining_loader(
            config.data,
            batch_size=config.micro_batch_size,
            rank=self.rank,
            world_size=self.world_size,
        )

    def _setup_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _build_model(self) -> torch.nn.Module:
        model = MoEModel(self.config.model)
        if torch.cuda.is_available():
            model = model.to(self.device)
        if is_distributed():
            model = wrap_with_fsdp(
                model,
                config=FSDPConfig(
                    sharding_strategy="FULL_SHARD",
                    backward_prefetch="BACKWARD_PRE",
                    activation_checkpointing=True,
                    mixed_precision=True,
                    use_orig_params=True,
                    transformer_block_classes=(
                        "MoEDecoderLayer",
                        "TitanDecoderLayer",
                    ),
                ),
                precision=self.precision_policy.precision,
                device_id=self.local_rank if torch.cuda.is_available() else None,
            )
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

    def _lr_at(self, step: int) -> float:
        warmup = max(1, self.config.warmup_steps)
        total = max(1, self.config.total_steps)
        if step < warmup:
            return self.config.learning_rate * (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.config.min_learning_rate + (
            self.config.learning_rate - self.config.min_learning_rate
        ) * cosine

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train(self) -> dict:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self.model.train()
        global_step = 0
        micro_idx = 0
        accumulation_loss = 0.0
        loader_iter = iter(self.loader)
        history = {
            "loss": [],
            "grad_norm": [],
            "expert_cov": [],
            "expert_histogram": [],
            "aux_loss": [],
            "z_loss": [],
            "lr": [],
        }

        while global_step < self.config.total_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                batch = next(loader_iter)

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            is_last_micro = micro_idx + 1 == self.config.gradient_accumulation_steps

            with accumulate_grad_steps(self.model, is_last_microstep=is_last_micro):
                with self.precision_policy.autocast():
                    _, loss, stats = self.model(input_ids, labels=labels)
                    loss = loss / self.config.gradient_accumulation_steps
                self.precision_policy.backward(loss)

            accumulation_loss += loss.detach().float().item()
            micro_idx += 1

            if not is_last_micro:
                continue

            self._set_lr(self._lr_at(global_step))
            record = self.precision_policy.step(
                self.optimizer,
                self.model.parameters(),
                step=global_step,
            )
            self.optimizer.zero_grad(set_to_none=True)
            micro_idx = 0

            history["loss"].append(accumulation_loss)
            history["grad_norm"].append(record.post_clip_norm)
            history["expert_cov"].append(stats.cov)
            history["expert_histogram"].append(stats.token_counts.tolist())
            history["aux_loss"].append(float(stats.aux_loss.item()))
            history["z_loss"].append(float(stats.z_loss.item()))
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if is_main_process() and global_step % self.config.log_every == 0:
                logger.info(
                    "step=%d loss=%.4f grad_norm=%.4f expert_cov=%.4f aux=%.4f z=%.4f lr=%.2e",
                    global_step,
                    accumulation_loss,
                    record.post_clip_norm,
                    stats.cov,
                    float(stats.aux_loss.item()),
                    float(stats.z_loss.item()),
                    self.optimizer.param_groups[0]["lr"],
                )

            if (
                self.config.checkpoint_every > 0
                and global_step > 0
                and global_step % self.config.checkpoint_every == 0
            ):
                save_distributed_checkpoint(
                    self.model,
                    self.optimizer,
                    self.config.output_dir,
                    step=global_step,
                    extra_state={"config": asdict(self.config)},
                )

            global_step += 1
            accumulation_loss = 0.0

        save_distributed_checkpoint(
            self.model,
            self.optimizer,
            self.config.output_dir,
            step=global_step,
            extra_state={"config": asdict(self.config)},
        )
        destroy_distributed()
        return history


__all__ = [
    "MoEPreTrainer",
    "MoEPreTrainerConfig",
]
