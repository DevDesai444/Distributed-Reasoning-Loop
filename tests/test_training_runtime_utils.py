"""
Tests for adapter-aware model loading during continued training.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.dpo_trainer import DPOTrainerConfig, ReasoningDPOTrainer
from training.runtime_utils import load_causal_lm_for_training


class _FakeModel:
    def __init__(self):
        self._params = {
            "base.weight": types.SimpleNamespace(requires_grad=True),
            "lora_A.weight": types.SimpleNamespace(requires_grad=True),
        }

    def named_parameters(self):
        return list(self._params.items())


def test_load_causal_lm_for_training_detects_adapter_checkpoint(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter_model"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")

    calls = {"base_model_name": None, "adapter_path": None}

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["base_model_name"] = model_name
            return _FakeModel()

    class FakePeftConfig:
        @staticmethod
        def from_pretrained(path):
            assert path == str(adapter_dir)
            return types.SimpleNamespace(base_model_name_or_path="base-model")

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, path, is_trainable=True):
            calls["adapter_path"] = path
            assert is_trainable is True
            return base_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(PeftConfig=FakePeftConfig, PeftModel=FakePeftModel),
    )

    model, use_8bit, loaded_adapter, base_model_name = load_causal_lm_for_training(
        str(adapter_dir),
        prefer_bf16=False,
        allow_8bit=False,
    )

    assert isinstance(model, _FakeModel)
    assert use_8bit is False
    assert loaded_adapter is True
    assert base_model_name == "base-model"
    assert calls["base_model_name"] == "base-model"
    assert calls["adapter_path"] == str(adapter_dir)


def test_dpo_train_preserves_non_trl_import_errors(monkeypatch):
    trainer = ReasoningDPOTrainer(DPOTrainerConfig())

    monkeypatch.setitem(
        sys.modules,
        "trl",
        types.SimpleNamespace(DPOTrainer=object, DPOConfig=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=lambda items: items)),
    )
    monkeypatch.setattr(trainer, "setup", lambda: (_ for _ in ()).throw(ImportError("adapter mismatch")))

    with pytest.raises(ImportError, match="adapter mismatch"):
        trainer.train([{"prompt": "p", "chosen": "c", "rejected": "r"}])


def test_dpo_train_filters_unsupported_trl_kwargs(monkeypatch):
    trainer = ReasoningDPOTrainer(
        DPOTrainerConfig(
            output_dir="./tmp-dpo-output",
            max_length=256,
            max_prompt_length=64,
        )
    )

    captured = {}

    class FakeDPOConfig:
        def __init__(
            self,
            output_dir,
            num_train_epochs,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            gradient_accumulation_steps,
            learning_rate,
            warmup_ratio,
            weight_decay,
            max_grad_norm,
            logging_steps,
            eval_steps,
            save_steps,
            evaluation_strategy,
            fp16,
            bf16,
            beta,
            loss_type,
            max_length,
            remove_unused_columns,
        ):
            captured["config_kwargs"] = {
                "output_dir": output_dir,
                "evaluation_strategy": evaluation_strategy,
                "max_length": max_length,
                "remove_unused_columns": remove_unused_columns,
            }

    class FakeDPOTrainer:
        def __init__(self, model, ref_model, args, train_dataset, eval_dataset, tokenizer):
            captured["trainer_kwargs"] = {
                "model": model,
                "ref_model": ref_model,
                "args": args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "tokenizer": tokenizer,
            }

        def train(self):
            captured["train_called"] = True

    monkeypatch.setitem(
        sys.modules,
        "trl",
        types.SimpleNamespace(DPOTrainer=FakeDPOTrainer, DPOConfig=FakeDPOConfig),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=lambda items: items)),
    )

    trainer.model = object()
    trainer.ref_model = None
    trainer.tokenizer = object()
    trainer.setup = lambda: None
    trainer.save = lambda path=None: None

    trainer.train([{"prompt": "p", "chosen": "c", "rejected": "r"}], eval_data=[{"prompt": "p", "chosen": "c", "rejected": "r"}])

    assert captured["config_kwargs"]["evaluation_strategy"] == "steps"
    assert captured["config_kwargs"]["max_length"] == 256
    assert "trainer_kwargs" in captured
    assert captured["trainer_kwargs"]["tokenizer"] is trainer.tokenizer
    assert captured["train_called"] is True
