"""I/O and inference round-trip for ShortcutPRM."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _build_tiny_prm():
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    config = BertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=30522,
        max_position_embeddings=128,
        type_vocab_size=2,
        num_labels=2,
    )
    backbone = BertForSequenceClassification(config)
    try:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)
        tokenizer.model_max_length = 128
    except Exception:
        from training.shortcut_prm_trainer import _build_smoke_tokenizer

        tokenizer = _build_smoke_tokenizer(model_max_length=128)
    from training.shortcut_prm import ShortcutPRM, ShortcutPRMConfig

    cfg = ShortcutPRMConfig(model_name="test-bert", max_length=128, num_labels=2)
    return ShortcutPRM(config=cfg, tokenizer=tokenizer, backbone=backbone)


def test_score_returns_probability_in_unit_interval():
    prm = _build_tiny_prm()
    score = prm.score("Step 1: trivial trace.")
    assert 0.0 <= score <= 1.0


def test_score_batch_lengths_match():
    prm = _build_tiny_prm()
    traces = ["one", "two", "three"]
    scores = prm.score_batch(traces)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_save_and_reload_roundtrip(tmp_path):
    pytest.importorskip("transformers")
    try:
        from transformers import BertTokenizerFast

        BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)
    except Exception:
        pytest.skip("HF BERT tokenizer not cached; skipping save/load roundtrip")

    prm = _build_tiny_prm()
    out = tmp_path / "prm_checkpoint"
    prm.save_pretrained(out)
    assert (out / "shortcut_prm_config.json").exists()

    from training.shortcut_prm import ShortcutPRM

    reloaded = ShortcutPRM.load(out)
    assert reloaded.config.num_labels == 2
    assert 0.0 <= reloaded.score("test trace") <= 1.0
