"""Smoke tests for the Layer-0 pre-training stack."""

from __future__ import annotations

import os

import torch

from src.training.pretraining.data import (
    PretrainingDataConfig,
    StreamingTokenizedDataset,
    synthetic_token_shards,
)
from src.training.pretraining.model import (
    TitanModel,
    TitanModelConfig,
    compute_mfu,
)


def _tiny_model_config(vocab_size: int = 256, seq_len: int = 32) -> TitanModelConfig:
    return TitanModelConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
    )


def test_titan_forward_returns_logits_and_loss():
    config = _tiny_model_config()
    model = TitanModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    labels = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    logits, loss = model(input_ids, labels=labels)
    assert logits.shape == (2, config.max_seq_len, config.vocab_size)
    assert loss.requires_grad
    assert loss.dim() == 0


def test_titan_backward_runs_a_step():
    config = _tiny_model_config()
    model = TitanModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
    labels = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
    _, loss = model(input_ids, labels=labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(loss)


def test_streaming_dataset_yields_pairs(tmp_path):
    synthetic_token_shards(
        str(tmp_path), num_shards=2, tokens_per_shard=128, vocab_size=200, seed=1
    )
    config = PretrainingDataConfig(
        data_dir=str(tmp_path), seq_len=32, file_glob="shard-*.jsonl"
    )
    dataset = StreamingTokenizedDataset(config)
    samples = []
    for sample in dataset:
        samples.append(sample)
        if len(samples) >= 3:
            break
    assert len(samples) >= 1
    assert samples[0]["input_ids"].shape == (32,)
    assert samples[0]["labels"].shape == (32,)


def test_compute_mfu_positive_for_simple_input():
    config = _tiny_model_config()
    mfu = compute_mfu(config, tokens_per_second=1000.0, peak_tflops=10.0)
    assert mfu >= 0.0


def test_streaming_dataset_partitions_by_rank(tmp_path):
    synthetic_token_shards(
        str(tmp_path), num_shards=4, tokens_per_shard=64, vocab_size=200, seed=2
    )
    config = PretrainingDataConfig(
        data_dir=str(tmp_path), seq_len=32, file_glob="shard-*.jsonl"
    )
    rank0 = StreamingTokenizedDataset(config, rank=0, world_size=2)
    rank1 = StreamingTokenizedDataset(config, rank=1, world_size=2)
    files0 = rank0._own_shards()
    files1 = rank1._own_shards()
    assert files0 and files1
    assert set(files0).isdisjoint(set(files1))
