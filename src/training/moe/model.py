"""
Mixture-of-Experts decoder for the DRL training stack.

Implements a small MoE decoder with:

  - 8 experts and top-2 routing (configurable).
  - SwiGLU experts driven by a learned softmax router.
  - Expert-parallel placement on top of FSDP for non-expert parameters.
  - Auxiliary load-balancing loss (Switch Transformer-style) with a tunable
    coefficient.
  - Per-step expert-utilization histograms and coefficient-of-variation
    statistics, exposed via :class:`RoutingStats` so the trainer can log
    routing imbalance live.

The non-MoE parts (attention, normalization, embeddings) reuse the Titan
decoder building blocks to keep the architecture, init, and FSDP wrap policy
consistent across Layer 0 dense and Layer 0 MoE pre-training.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pretraining.model import (
    RMSNorm,
    TitanAttention,
    TitanModelConfig,
    _build_rope_cache,
)


@dataclass
class MoEConfig(TitanModelConfig):
    """Configuration for the MoE decoder."""

    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 2048
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 1.0e-3
    capacity_factor: float = 1.25
    expert_parallel_size: int = 1


@dataclass
class RoutingStats:
    """Routing telemetry produced during a forward pass.

    Attributes:
        token_counts: tokens routed to each expert across all MoE layers.
        cov: coefficient of variation of token counts (load imbalance).
        aux_loss: auxiliary load-balancing loss aggregated over layers.
        z_loss: router log-sum-exp regularizer aggregated over layers.
        per_layer_cov: per-layer CoV for diagnostics.
    """

    token_counts: torch.Tensor
    cov: float
    aux_loss: torch.Tensor
    z_loss: torch.Tensor
    per_layer_cov: List[float] = field(default_factory=list)


class MoESwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoERouter(nn.Module):
    """Top-k softmax router with Switch-style auxiliary loss."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(probs, k=self.config.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1.0e-9)

        mask = torch.zeros_like(probs)
        mask.scatter_(-1, top_idx, 1.0)
        token_share = mask.mean(dim=tuple(range(mask.ndim - 1)))
        prob_share = probs.mean(dim=tuple(range(probs.ndim - 1)))
        aux_loss = (token_share * prob_share).sum() * self.config.n_experts
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        return top_idx, top_w, aux_loss, z_loss, mask


class MoEMixtureLayer(nn.Module):
    """Top-k MoE FFN."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.router = MoERouter(config)
        local_experts = self._owned_expert_count()
        self.expert_offset = self._expert_offset()
        self.experts = nn.ModuleList(
            [
                MoESwiGLUExpert(config.d_model, config.expert_hidden)
                for _ in range(local_experts)
            ]
        )

    def _owned_expert_count(self) -> int:
        if self.config.expert_parallel_size <= 1:
            return self.config.n_experts
        per_rank = self.config.n_experts // self.config.expert_parallel_size
        rank = int(os.environ.get("LOCAL_RANK", "0")) % self.config.expert_parallel_size
        extra = 1 if rank < self.config.n_experts % self.config.expert_parallel_size else 0
        return per_rank + extra

    def _expert_offset(self) -> int:
        if self.config.expert_parallel_size <= 1:
            return 0
        per_rank = self.config.n_experts // self.config.expert_parallel_size
        rank = int(os.environ.get("LOCAL_RANK", "0")) % self.config.expert_parallel_size
        extra = min(rank, self.config.n_experts % self.config.expert_parallel_size)
        return rank * per_rank + extra

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, d_model = x.shape
        flat = x.reshape(-1, d_model)
        top_idx, top_w, aux_loss, z_loss, mask = self.router(x)
        top_idx = top_idx.reshape(-1, self.config.top_k)
        top_w = top_w.reshape(-1, self.config.top_k)

        output = torch.zeros_like(flat)
        token_counts = torch.zeros(
            self.config.n_experts, device=flat.device, dtype=torch.float32
        )
        for slot in range(self.config.top_k):
            expert_ids = top_idx[:, slot]
            weights = top_w[:, slot].unsqueeze(-1)
            for local_idx, expert in enumerate(self.experts):
                global_id = local_idx + self.expert_offset
                token_mask = expert_ids == global_id
                if not token_mask.any():
                    continue
                selected = flat[token_mask]
                processed = expert(selected) * weights[token_mask]
                output[token_mask] = output[token_mask] + processed
                token_counts[global_id] = token_counts[global_id] + float(token_mask.sum())

        return output.view(bsz, seq_len, d_model), aux_loss, z_loss, token_counts


class MoEDecoderLayer(nn.Module):
    """One transformer block where the FFN is replaced by an MoE.

    Named explicitly so :mod:`src.training.fsdp_utils` can target it from the
    transformer auto-wrap policy.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model, eps=config.rms_eps)
        self.attn = TitanAttention(config)
        self.norm_mlp = RMSNorm(config.d_model, eps=config.rms_eps)
        self.moe = MoEMixtureLayer(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.attn(self.norm_attn(x), cos, sin, attention_mask))
        moe_out, aux_loss, z_loss, token_counts = self.moe(self.norm_mlp(x))
        x = x + self.dropout(moe_out)
        return x, aux_loss, z_loss, token_counts


class MoEModel(nn.Module):
    """Causal LM with MoE FFN blocks and routing telemetry."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [MoEDecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model, eps=config.rms_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tied_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = _build_rope_cache(config.head_dim(), config.max_seq_len, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        x = self.embed(input_ids)
        aux_total = torch.zeros((), device=x.device)
        z_total = torch.zeros((), device=x.device)
        per_layer_counts: List[torch.Tensor] = []
        per_layer_cov: List[float] = []

        for layer in self.layers:
            x, aux_loss, z_loss, token_counts = layer(
                x, self.rope_cos, self.rope_sin, attention_mask
            )
            aux_total = aux_total + aux_loss
            z_total = z_total + z_loss
            per_layer_counts.append(token_counts)
            per_layer_cov.append(_coefficient_of_variation(token_counts))

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = (
                ce
                + self.config.aux_loss_coef * aux_total
                + self.config.router_z_loss_coef * z_total
            )

        token_counts = torch.stack(per_layer_counts).sum(dim=0)
        stats = RoutingStats(
            token_counts=token_counts.detach(),
            cov=_coefficient_of_variation(token_counts.detach()),
            aux_loss=aux_total.detach(),
            z_loss=z_total.detach(),
            per_layer_cov=per_layer_cov,
        )
        return logits, loss, stats

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _coefficient_of_variation(counts: torch.Tensor) -> float:
    counts = counts.float()
    mean = counts.mean()
    if mean.item() <= 0.0:
        return 0.0
    std = counts.std(unbiased=False)
    return float(std.item() / mean.item())


__all__ = [
    "MoEConfig",
    "MoEDecoderLayer",
    "MoEModel",
    "MoERouter",
    "MoEMixtureLayer",
    "MoESwiGLUExpert",
    "RoutingStats",
]
