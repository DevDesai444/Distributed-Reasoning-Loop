"""
Titan decoder — the DRL Layer 0 base model.

A torchtitan-style Llama decoder: pre-norm transformer with rotary position
embeddings, SwiGLU MLP, RMSNorm, grouped-query attention, and tied I/O
embeddings. Default configuration targets ~250M parameters (12 layers,
d_model 1024, 16 heads) — the size the report engineers around so a full
pre-training run is reproducible on 4 × A100-80GB inside one day.

The module is deliberately framework-agnostic: it is plain ``torch.nn`` and
plays cleanly with FSDP's transformer auto-wrap policy because the per-block
class is exported as :class:`TitanDecoderLayer`, the same name registered in
:mod:`src.training.fsdp_utils`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TitanModelConfig:
    """Architecture hyperparameters for the DRL Layer-0 base model."""

    vocab_size: int = 50_304
    max_seq_len: int = 2048
    n_layers: int = 12
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 8
    ffn_hidden: int = 2816
    rope_base: float = 10_000.0
    rms_eps: float = 1.0e-5
    tied_embeddings: bool = True
    dropout: float = 0.0
    initializer_range: float = 0.02

    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}."
            )
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype) * self.weight)


def _build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotated * sin


class TitanAttention(nn.Module):
    def __init__(self, config: TitanModelConfig):
        super().__init__()
        self.config = config
        head_dim = config.head_dim()
        self.head_dim = head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos_slice = cos[:seq_len]
        sin_slice = sin[:seq_len]
        cos_slice = torch.cat((cos_slice, cos_slice), dim=-1)
        sin_slice = torch.cat((sin_slice, sin_slice), dim=-1)
        q = _apply_rope(q, cos_slice, sin_slice)
        k = _apply_rope(k, cos_slice, sin_slice)

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TitanSwiGLU(nn.Module):
    def __init__(self, config: TitanModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.w_up = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.w_down = nn.Linear(config.ffn_hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TitanDecoderLayer(nn.Module):
    """One transformer block. Named so FSDP's auto-wrap policy targets it."""

    def __init__(self, config: TitanModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model, eps=config.rms_eps)
        self.attn = TitanAttention(config)
        self.norm_mlp = RMSNorm(config.d_model, eps=config.rms_eps)
        self.mlp = TitanSwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm_attn(x), cos, sin, attention_mask))
        x = x + self.dropout(self.mlp(self.norm_mlp(x)))
        return x


class TitanModel(nn.Module):
    """Causal LM with tied embeddings, RoPE, RMSNorm, SwiGLU, GQA."""

    def __init__(self, config: TitanModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [TitanDecoderLayer(config) for _ in range(config.n_layers)]
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
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_mfu(
    config: TitanModelConfig,
    tokens_per_second: float,
    peak_tflops: float,
) -> float:
    """Model FLOPs Utilization for a single training step.

    Uses the Chinchilla 6N FLOPs-per-token approximation augmented with a
    rough attention term, normalized by peak device FLOPS in TF/s.
    """
    n = sum(
        param.numel()
        for param in TitanModel(config).parameters()
        if param.requires_grad
    )
    seq = config.max_seq_len
    flops_per_token = 6.0 * n + 12.0 * config.n_layers * config.d_model * seq
    flops_per_sec = tokens_per_second * flops_per_token
    return flops_per_sec / (peak_tflops * 1.0e12) if peak_tflops > 0 else 0.0


__all__ = [
    "TitanModelConfig",
    "TitanModel",
    "TitanDecoderLayer",
    "compute_mfu",
]
