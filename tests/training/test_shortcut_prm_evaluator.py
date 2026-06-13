"""PRM evaluator wrapper test."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _build_prm():
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        vocab_size=30522,
        max_position_embeddings=64,
        type_vocab_size=2,
        num_labels=2,
    )
    backbone = BertForSequenceClassification(cfg)
    from training.shortcut_prm_trainer import _build_smoke_tokenizer
    from training.shortcut_prm import ShortcutPRM, ShortcutPRMConfig

    tokenizer = _build_smoke_tokenizer(model_max_length=64)
    return ShortcutPRM(
        config=ShortcutPRMConfig(model_name="test", max_length=64, num_labels=2),
        tokenizer=tokenizer,
        backbone=backbone,
    )


def test_prm_evaluator_returns_unit_interval_scores():
    from evaluation.judges.prm_evaluator import PRMEvaluator

    prm = _build_prm()
    evaluator = PRMEvaluator(prm)
    scores = evaluator.score_batch(["step 1 trace", "no derivation", "another one"])
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_prm_evaluator_judge_verdicts_have_required_fields():
    from evaluation.judges.prm_evaluator import PRMEvaluator
    from evaluation.judges import Verdict

    prm = _build_prm()
    evaluator = PRMEvaluator(prm)
    verdicts = evaluator.judge_verdicts(["alpha", "beta"])
    assert len(verdicts) == 2
    for v in verdicts:
        assert v.verdict in {Verdict.ACCEPT, Verdict.REJECT, Verdict.UNCERTAIN}
        assert 0.0 <= v.confidence <= 1.0
        assert v.judge_id == "shortcut-prm"
        assert v.prompt_version


def test_prm_evaluator_threshold_flip():
    from evaluation.judges.prm_evaluator import PRMEvaluator
    from evaluation.judges import Verdict

    prm = _build_prm()
    evaluator = PRMEvaluator(prm)
    # threshold=0 -> nothing scores below 0 -> all REJECT
    verdicts_low = evaluator.judge_verdicts(["x", "y"], shortcut_threshold=0.0)
    # threshold=1 -> everything below 1 -> all ACCEPT
    verdicts_high = evaluator.judge_verdicts(["x", "y"], shortcut_threshold=1.0)
    assert all(v.verdict == Verdict.REJECT for v in verdicts_low)
    assert all(v.verdict == Verdict.ACCEPT for v in verdicts_high)
