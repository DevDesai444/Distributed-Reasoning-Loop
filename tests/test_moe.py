"""Smoke tests for the MoE training variant."""

from __future__ import annotations

import torch

from src.training.moe.model import MoEConfig, MoEModel


def _tiny_moe_config(vocab_size: int = 256, seq_len: int = 32) -> MoEConfig:
    return MoEConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        n_experts=4,
        top_k=2,
        expert_hidden=64,
    )


def test_moe_forward_returns_logits_loss_and_stats():
    config = _tiny_moe_config()
    model = MoEModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    labels = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    logits, loss, stats = model(input_ids, labels=labels)
    assert logits.shape == (2, config.max_seq_len, config.vocab_size)
    assert loss.requires_grad
    assert stats.token_counts.numel() == config.n_experts
    assert 0.0 <= stats.cov <= 5.0
    assert len(stats.per_layer_cov) == config.n_layers


def test_moe_backward_runs_a_step():
    config = _tiny_moe_config()
    model = MoEModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
    labels = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
    _, loss, _ = model(input_ids, labels=labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(loss)


def test_routing_stats_reflect_token_distribution():
    config = _tiny_moe_config()
    model = MoEModel(config)
    input_ids = torch.randint(0, config.vocab_size, (4, config.max_seq_len))
    _, _, stats = model(input_ids)
    assert stats.token_counts.sum() > 0
